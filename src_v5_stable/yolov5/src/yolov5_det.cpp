#include "cuda_utils.h"
#include "logging.h"
#include "utils.h"
#include "preprocess.h"
#include "postprocess.h"
#include "model.h"
#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <opencv2/highgui/highgui.hpp>
#include <cv_bridge/cv_bridge.h>
#include <std_msgs/UInt8.h>

#include "result.h"
#include "pre_datasets.h"
#include <iostream>
#include <chrono>
#include <cmath>

using namespace nvinfer1;

static Logger gLogger;
const static int kOutputSize = kMaxNumOutputBbox * sizeof(Detection) / sizeof(float) + 1;

int count_= 0;

struct Result_buffer{
    int x;
    int y;
    int distance;
    int height;
    float confidence;
};

struct Box_recorder{
    int img_count;
    float class_id;
    float cx;
    float cy;
    float w;
    float h;
};
Box_recorder box_recorder = {0, 0, 0, 0, 0, 0};
yolov5::pre_datasets box_record_msg;
ros::Publisher boxPub;

Result_buffer result_buffer[4] = {
        {0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0}
};

int direction_buffer = -1;
int aim_zero_count = 0;

cv::Mat detected_src;
cv::Mat origin_src;

float* gpu_buffers[2];
cudaStream_t stream;
IRuntime* runtime = nullptr;
ICudaEngine* engine = nullptr;
IExecutionContext* context = nullptr;
float* cpu_output_buffer = nullptr;

// 調參區
int focal_length = 608;
int height_limit_basket = 16;
int height_limit_volley = 13;
bool if_show_in_cv = true, if_show_in_ros = false, if_save = false;
bool if_show_all_rect = true, if_record_box = false;
uint8_t aim = 0;


void prepare_buffers(ICudaEngine* engine, float** gpu_input_buffer, float** gpu_output_buffer, float** cpu_output_buffer) {
    assert(engine->getNbBindings() == 2);
    // In order to bind the buffers, we need to know the names of the input and output tensors.
    // Note that indices are guaranteed to be less than IEngine::getNbBindings()
    const int inputIndex = engine->getBindingIndex(kInputTensorName);
    const int outputIndex = engine->getBindingIndex(kOutputTensorName);
    assert(inputIndex == 0);
    assert(outputIndex == 1);
    // Create GPU buffers on device
    CUDA_CHECK(cudaMalloc((void**)gpu_input_buffer, kBatchSize * 3 * kInputH * kInputW * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)gpu_output_buffer, kBatchSize * kOutputSize * sizeof(float)));

    *cpu_output_buffer = new float[kBatchSize * kOutputSize];
}

void infer(IExecutionContext& context, cudaStream_t& stream, void** gpu_buffers, float* output, int batchsize) {
    context.enqueue(batchsize, gpu_buffers, stream, nullptr);
    CUDA_CHECK(cudaMemcpyAsync(output, gpu_buffers[1], batchsize * kOutputSize * sizeof(float), cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);
}

void serialize_engine(unsigned int max_batchsize, bool& is_p6, float& gd, float& gw, std::string& wts_name, std::string& engine_name) {
    // Create builder
    IBuilder* builder = createInferBuilder(gLogger);
    IBuilderConfig* config = builder->createBuilderConfig();

    // Create model to populate the network, then set the outputs and create an engine
    ICudaEngine *engine = nullptr;
    if (is_p6) {
    engine = build_det_p6_engine(max_batchsize, builder, config, DataType::kFLOAT, gd, gw, wts_name);
    } else {
    engine = build_det_engine(max_batchsize, builder, config, DataType::kFLOAT, gd, gw, wts_name);
    }
    assert(engine != nullptr);

    // Serialize the engine
    IHostMemory* serialized_engine = engine->serialize();
    assert(serialized_engine != nullptr);

    // Save engine to file
    std::ofstream p(engine_name, std::ios::binary);
    if (!p) {
    std::cerr << "Could not open plan output file" << std::endl;
    assert(false);
    }
    p.write(reinterpret_cast<const char*>(serialized_engine->data()), serialized_engine->size());

    // Close everything down
    engine->destroy();
    builder->destroy();
    config->destroy();
    serialized_engine->destroy();
}

void deserialize_engine(std::string& engine_name, IRuntime** runtime, ICudaEngine** engine, IExecutionContext** context) {
    std::ifstream file(engine_name, std::ios::binary);
    if (!file.good()) {
    std::cerr << "read " << engine_name << " error!" << std::endl;
    assert(false);
    }
    size_t size = 0;
    file.seekg(0, file.end);
    size = file.tellg();
    file.seekg(0, file.beg);
    char* serialized_engine = new char[size];
    assert(serialized_engine);
    file.read(serialized_engine, size);
    file.close();

    *runtime = createInferRuntime(gLogger);
    assert(*runtime);
    *engine = (*runtime)->deserializeCudaEngine(serialized_engine, size);
    assert(*engine);
    *context = (*engine)->createExecutionContext();
    assert(*context);
    delete[] serialized_engine;
}

void calc_distance(std::vector<cv::Mat> &img_batch, std::vector<std::vector<Detection>> &res_batch) {
    count_ ++;
    if (res_batch[0].size() == 0){
        // no detected object
        direction_buffer = -1;
        if (if_record_box) {
            cv::Mat merge_img, tmp_img = img_batch[0];
            // cv::hconcat(tmp_img, tmp_img, merge_img);
            char datasets_name[256], check_name[256];   //图像保存
            sprintf(datasets_name,"/home/nuaa/Downloads/hlf_files/datasets_img/%d.jpg",count_);
            // sprintf(check_name,"/home/nuaa/Downloads/hlf_files/check_img/%d.jpg",count_);
            // cv::imwrite(check_name, merge_img);
            cv::imwrite(datasets_name, tmp_img);
            ROS_WARN("box: %d", count_);
        }
    }
    else{
        // find object
        std::vector<Result_buffer> result_temp_list[4];
        for (size_t i = 0; i < img_batch.size(); i++) {
            if (i) break;
            auto &res = res_batch[i];

            // // remove repeat box
            // for (auto & Ta : res){
            //     if (Ta.class_id == 0.0f || Ta.class_id == 1.0f){
            //         for (auto & Tb : res){
            //             if (Tb.class_id == 0.0f || Tb.class_id == 1.0f){
            //                 float similarity = (Tb.bbox[0] / Ta.bbox[0] + Tb.bbox[1] / Ta.bbox[1] 
            //                 + Tb.bbox[2] * Tb.bbox[3] / Ta.bbox[2] / Ta.bbox[3]) / 3;
            //                 if (Tb.bbox[2] * Tb.bbox[3] != Ta.bbox[2] * Ta.bbox[3] && 
            //                 similarity < 1.1f && similarity > 0.9f){
            //                     if (Ta.conf > Tb.conf){
            //                         Tb.class_id = 199.0f;
            //                     } else{
            //                         Ta.class_id = 199.0f;
            //                     }
            //                 }
            //             }
            //         }
            //     }
            // }

            cv::Mat img = img_batch[i];
            if (if_record_box) img.copyTo(origin_src);
            for (size_t j = 0; j < res.size(); j++) {
                cv::Rect r = get_rect(img, res[j].bbox);
                if (if_record_box && (res[j].class_id == 0.0f || res[j].class_id == 1.0f
                 || res[j].class_id == 2.0f || res[j].class_id == 3.0f)) {
                    box_record_msg.img_count = count_;
                    box_record_msg.class_id = res[j].class_id;
                    box_record_msg.cx = res[j].bbox[0] / 640;
                    box_record_msg.cy = res[j].bbox[1] / 480;
                    box_record_msg.w = res[j].bbox[2] / 640;
                    box_record_msg.h = res[j].bbox[3] / 480;
                    boxPub.publish(box_record_msg);
                    ROS_WARN("box: %d", count_);
                    // box_recorder = (Box_recorder){count_, res[j].class_id, 
                    // res[j].bbox[0], res[j].bbox[1], res[j].bbox[2], res[j].bbox[3]}
                }
                // real_distance / real_width = focal_distance / pixel_width
                // height / real_distance = pixel_height / focal_length
                int distance = 0, height = 0;
                float confidence = .0f;

                if ((int)res[j].class_id == 0 || (int)res[j].class_id == 1) {
                    float real_width = 20 + 4.5 * (int)res[j].class_id;
                    int calc_w = 0;
                    calc_w = r.width > r.height ? r.width : r.height;
                    distance = real_width * focal_length / calc_w;
                    height = distance * (240 - r.y - r.height / 2) / focal_length;
                    confidence = res[j].conf;
                }
                else if((int)res[j].class_id == 3) {
                    int calc_w = r.width > r.height ? r.height : r.width;
                    distance = int(20 * focal_length / calc_w);
                    confidence = res[j].conf;
                }

                if (res[j].class_id <= 3)
                result_temp_list[(int)res[j].class_id].emplace_back(
                        (Result_buffer){r.tl().x + r.width/2, r.tl().y + r.height/2,
                                        distance, height, confidence});

                if (if_show_all_rect){
                // if (res[j].class_id == 199.0f){
                    cv::rectangle(img, r, cv::Scalar(0x27, 0xC1, 0x36), 2);
                    cv::putText(img, std::to_string((int) res[j].class_id), cv::Point(r.x, r.y - 1), cv::FONT_HERSHEY_PLAIN,
                                1.2, cv::Scalar(0x00, 0xFF, 0x00), 1);
                    cv::putText(img, std::to_string(distance), cv::Point(r.x, r.y + 15), cv::FONT_HERSHEY_PLAIN,
                                1.2, cv::Scalar(0xFF, 0x00, 0xFF), 1);
                    cv::putText(img, std::to_string(height), cv::Point(r.x + 30, r.y + 15), cv::FONT_HERSHEY_PLAIN,
                                1.2, cv::Scalar(0xFF, 0x00, 0xFF), 1);
                }
            }
            if (if_show_all_rect) detected_src = img;
            if (if_record_box) {
                // cv::Mat merge_img;
                // cv::hconcat(detected_src, origin_src, merge_img);
                char datasets_name[256], check_name[256];   //图像保存
                sprintf(datasets_name,"/home/nuaa/Downloads/hlf_files/datasets_img/%d.jpg",count_);
                // sprintf(check_name,"/home/nuaa/Downloads/hlf_files/check_img/%d.jpg",count_);
                // cv::imwrite(check_name, merge_img);
                cv::imwrite(datasets_name, origin_src);
            }
        }

        // yolov8 label 0:volleyball, 1:basketball, 2:hoop, 3:column
        // mode 0:none, 1:volleyball, 2:basketball, 3:column, 4:hoop
        switch (aim) {
            case 1:{
                if (result_temp_list[0].empty()){
                    result_buffer[0] = (Result_buffer){0, 0, 0, 0, 0};
                    direction_buffer = -1;
                }
                else{
                    auto target_iter = result_temp_list[0].begin();
                    int min_dist = 10000;
                    float max_confidence = 0;
                    for (auto iter = result_temp_list[0].begin(); iter != result_temp_list[0].end(); iter++){
                        if ((*iter).distance < min_dist && (*iter).height < height_limit_volley){
                            target_iter = iter;
                            min_dist = (*iter).distance;
                        }
                    }
                    result_buffer[0] = (*target_iter);
                    direction_buffer = 0;
                }
                break;
            }
            case 2:{
                if (result_temp_list[1].empty()){
                    result_buffer[1] = (Result_buffer){0, 0, 0, 0, 0};
                    direction_buffer = -1;
                }
                else{
                    auto target_iter = result_temp_list[1].begin();
                    int min_dist = 10000;
                    for (auto iter = result_temp_list[1].begin(); iter != result_temp_list[1].end(); iter++){
                        if ((*iter).distance < min_dist && (*iter).height < height_limit_basket){
                            target_iter = iter;
                            min_dist = (*iter).distance;
                        }
                    }
                    result_buffer[1] = (*target_iter);
                    direction_buffer = 0;
                }
                break;
            }
            case 4:{
                if (result_temp_list[3].empty()){
                    result_buffer[3] = (Result_buffer){0, 0, 0, 0, 0};
                    direction_buffer = -1;
                }
                else{
                    auto target_iter = result_temp_list[3].begin();
                    int max_dist = 0;
                    for (auto iter = result_temp_list[3].begin(); iter != result_temp_list[3].end(); iter++){
                        if ((*iter).distance > max_dist){
                            target_iter = iter;
                            max_dist = (*iter).distance;
                        }
                    }
                    result_buffer[3] = (*target_iter);
                    direction_buffer = 0;
                }
                break;
            }
            default:{
                direction_buffer = -1;
            }
        }
    }
}

void modeCallback(const std_msgs::UInt8::ConstPtr& msg)
{
    // mode 0:none, 1:volleyball, 2:basketball, 3:column, 4:hoop
    // yolov8 label 0:volleyball, 1:basketball, 2:hoop, 3:column
    aim = msg->data;
    if (aim == 3) {
        aim = 4;} 
    else if (aim == 4) {
        aim = 3;}
    ROS_INFO("detect aim : %d", aim);
}

void imageCallback(const sensor_msgs::ImageConstPtr& msg)
{
    cv::Mat src = cv_bridge::toCvShare(msg, "bgr8")->image;
    std::vector<cv::Mat> img_batch;
    img_batch.push_back(src);

     // Preprocess
    cuda_batch_preprocess(img_batch, gpu_buffers[0], kInputW, kInputH, stream);

    // Run inference
    infer(*context, stream, (void**)gpu_buffers, cpu_output_buffer, kBatchSize);

    // NMS
    std::vector<std::vector<Detection>> res_batch;
    batch_nms(res_batch, cpu_output_buffer, img_batch.size(), kOutputSize, kConfThresh, kNmsThresh);

    // Draw bounding boxes and calc distance height
    calc_distance(img_batch, res_batch);

    // Save detected images
    for (const auto & j : img_batch) {
        if (if_save){
            char base_name[256];   //图像保存
            sprintf(base_name,"/home/hlf/Downloads/myFiles/clac_focal_length/image/%04ld.jpg",count_++);
            cv::imwrite(base_name, j);
        }
        if (if_show_in_cv){
            cv::imshow("view", j);
        }
    }
}

int main(int argc, char** argv) {
    cudaSetDevice(kGpuId);

    ros::init(argc, argv, "yolov5");
    ros::NodeHandle nh;

    std::string wts_name;
    std::string engine_name;
    bool is_p6 = false;
    float gd = 0.0f, gw = 0.0f;

    char s_or_d = 'd';
    
    if (s_or_d == 'd'){
        engine_name = "/home/nuaa/Downloads/v5s_ws/src/yolov5/engine/10_12_8000.engine";
    } else if (s_or_d == 's'){
        wts_name = "/home/nuaa/Downloads/v5s_ws/src/yolov5/weights/10_12_8000.wts";
        engine_name = "/home/nuaa/Downloads/v5s_ws/src/yolov5/engine/10_12_8000.engine";
        gd = 0.33;
        gw = 0.50;
        is_p6 = false;
    }

    // Create a model using the API directly and serialize it to a file
    if (!wts_name.empty()) {
    serialize_engine(kBatchSize, is_p6, gd, gw, wts_name, engine_name);
    return 0;
    }

    // Deserialize the engine from file

    deserialize_engine(engine_name, &runtime, &engine, &context);
    CUDA_CHECK(cudaStreamCreate(&stream));

    // Init CUDA preprocessing
    cuda_preprocess_init(kMaxInputImageSize);

    // Prepare cpu and gpu buffers
    prepare_buffers(engine, &gpu_buffers[0], &gpu_buffers[1], &cpu_output_buffer);

    ros::Subscriber modeSub = nh.subscribe("/detect_mode", 1, &modeCallback);
    image_transport::ImageTransport it(nh);
    image_transport::Subscriber sub = it.subscribe("camera/color/image_raw", 1, &imageCallback);
    image_transport::Publisher detectedImgPub = it.advertise("/detected_img", 1);
    ros::Publisher resultPub = nh.advertise<yolov5::result>("/detect_result", 1);
    boxPub = nh.advertise<yolov5::pre_datasets>("/box_record", 1);
//    ros::Subscriber paramSub = nh.subscribe("/param", 1, &paramCallback);

    cv::namedWindow("view");
    cv::startWindowThread();

    ros::Rate loop_rate(500);
    while(ros::ok())
    {
        //发布检测后的图片
        if (if_show_in_ros){
            sensor_msgs::ImagePtr detected_img_msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", detected_src).toImageMsg();
            detectedImgPub.publish(detected_img_msg);
        }

        //发布检测结果
        yolov5::result result_msg;
        if (aim > 0 && aim < 5){
            result_msg.x = result_buffer[aim - 1].x;
            result_msg.y = result_buffer[aim - 1].y;
            result_msg.distance = result_buffer[aim - 1].distance;
        }
        result_msg.direction = direction_buffer;

        if (aim > 0 && aim < 5){
            resultPub.publish(result_msg);
        }

        ros::spinOnce();
        loop_rate.sleep();
    }

    cv::destroyWindow("view");

    // Release stream and buffers
    cudaStreamDestroy(stream);
    CUDA_CHECK(cudaFree(gpu_buffers[0]));
    CUDA_CHECK(cudaFree(gpu_buffers[1]));
    delete[] cpu_output_buffer;
    cuda_preprocess_destroy();
    // Destroy the engine
    context->destroy();
    engine->destroy();
    runtime->destroy();

    // Print histogram of the output distribution
    // std::cout << "\nOutput:\n\n";
    // for (unsigned int i = 0; i < kOutputSize; i++) {
    //   std::cout << prob[i] << ", ";
    //   if (i % 10 == 0) std::cout << std::endl;
    // }
    // std::cout << std::endl;

    return 0;
}


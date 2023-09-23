#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>
#include "utils.h"
#include "preprocess.h"
#include "postprocess.h"
#include "cuda_utils.h"
#include "logging.h"
#include <ros/ros.h>
#include <std_msgs/UInt8.h>
#include <image_transport/image_transport.h>
#include <opencv2/highgui/highgui.hpp>
#include <cv_bridge/cv_bridge.h>
#include "result.h"
#include "model.h"

using namespace nvinfer1;

long int count_=0;

Logger gLogger;

const int kOutputSize = kMaxNumOutputBbox * sizeof(Detection) / sizeof(float) + 1;

IRuntime *custom_runtime = nullptr;
ICudaEngine *custom_engine = nullptr;
IExecutionContext *custom_context = nullptr;
std::string custom_cuda_post_process = "g";
int custom_model_bboxes;

float *custom_device_buffers[2];
float *custom_output_buffer_host = nullptr;
float *custom_decode_ptr_host=nullptr;
float *custom_decode_ptr_device=nullptr;

cudaStream_t custom_stream;

enum target
{
    motionless, volleyball, basketball, column, hoop
};
target aim = basketball;

struct Result_buffer{
    int x;
    int y;
    int distance;
};

Result_buffer result_buffer[3] = {
        {0, 0, 0},
        {0, 0, 0},
        {0, 0, 0}
};

cv::Mat detected_src;
bool if_show = true, if_save = false;

//void paramCallback()

void serialize_engine(std::string &wts_name, std::string &engine_name, std::string &sub_type) {
    IBuilder *builder = createInferBuilder(gLogger);
    IBuilderConfig *config = builder->createBuilderConfig();
    IHostMemory *serialized_engine = nullptr;

    if (sub_type == "n") {
        serialized_engine = buildEngineYolov8n(builder, config, DataType::kFLOAT, wts_name);
    } else if (sub_type == "s") {
        serialized_engine = buildEngineYolov8s(builder, config, DataType::kFLOAT, wts_name);
    } else if (sub_type == "m") {
        serialized_engine = buildEngineYolov8m(builder, config, DataType::kFLOAT, wts_name);
    } else if (sub_type == "l") {
        serialized_engine = buildEngineYolov8l(builder, config, DataType::kFLOAT, wts_name);
    } else if (sub_type == "x") {
        serialized_engine = buildEngineYolov8x(builder, config, DataType::kFLOAT, wts_name);
    }

    assert(serialized_engine);
    std::ofstream p(engine_name, std::ios::binary);
    if (!p) {
        std::cout << "could not open plan output file" << std::endl;
        assert(false);
    }
    p.write(reinterpret_cast<const char *>(serialized_engine->data()), serialized_engine->size());

    delete builder;
    delete config;
    delete serialized_engine;
}

void deserialize_engine(std::string &engine_name, IRuntime **runtime, ICudaEngine **engine, IExecutionContext **context) {
    std::ifstream file(engine_name, std::ios::binary);
    if (!file.good()) {
        std::cerr << "read " << engine_name << " error!" << std::endl;
        assert(false);
    }
    size_t size = 0;
    file.seekg(0, std::ifstream::end);
    size = file.tellg();
    file.seekg(0, std::ifstream::beg);
    char *serialized_engine = new char[size];
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

void prepare_buffer(ICudaEngine *engine, float **input_buffer_device, float **output_buffer_device,
                    float **output_buffer_host, float **decode_ptr_host, float **decode_ptr_device, std::string cuda_post_process) {
    assert(engine->getNbBindings() == 2);
    // In order to bind the buffers, we need to know the names of the input and output tensors.
    // Note that indices are guaranteed to be less than IEngine::getNbBindings()
    const int inputIndex = engine->getBindingIndex(kInputTensorName);
    const int outputIndex = engine->getBindingIndex(kOutputTensorName);
    assert(inputIndex == 0);
    assert(outputIndex == 1);
    // Create GPU buffers on device
    CUDA_CHECK(cudaMalloc((void **) input_buffer_device, kBatchSize * 3 * kInputH * kInputW * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void **) output_buffer_device, kBatchSize * kOutputSize * sizeof(float)));
    if (cuda_post_process == "c") {
        *output_buffer_host = new float[kBatchSize * kOutputSize];
    } else if (cuda_post_process == "g") {
        if (kBatchSize > 1) {
            std::cerr << "Do not yet support GPU post processing for multiple batches" << std::endl;
            exit(0);
        }
        // Allocate memory for decode_ptr_host and copy to device
        *decode_ptr_host = new float[1 + kMaxNumOutputBbox * bbox_element];
        CUDA_CHECK(cudaMalloc((void **)decode_ptr_device, sizeof(float) * (1 + kMaxNumOutputBbox * bbox_element)));
    }
}

void infer(IExecutionContext &context, cudaStream_t &stream, void **buffers, float *output, int batchsize, float* decode_ptr_host, float* decode_ptr_device, int model_bboxes, std::string cuda_post_process) {
    // infer on the batch asynchronously, and DMA output back to host
    auto start = std::chrono::system_clock::now();
    context.enqueue(batchsize, buffers, stream, nullptr);
    if (cuda_post_process == "c") {
        CUDA_CHECK(cudaMemcpyAsync(output, buffers[1], batchsize * kOutputSize * sizeof(float), cudaMemcpyDeviceToHost,stream));
        auto end = std::chrono::system_clock::now();
        std::cout << "inference time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;
    } else if (cuda_post_process == "g") {
        CUDA_CHECK(cudaMemsetAsync(decode_ptr_device, 0, sizeof(float) * (1 + kMaxNumOutputBbox * bbox_element), stream));
        cuda_decode((float *)buffers[1], model_bboxes, kConfThresh, decode_ptr_device, kMaxNumOutputBbox, stream);
        cuda_nms(decode_ptr_device, kNmsThresh, kMaxNumOutputBbox, stream);//cuda nms
        CUDA_CHECK(cudaMemcpyAsync(decode_ptr_host, decode_ptr_device, sizeof(float) * (1 + kMaxNumOutputBbox * bbox_element), cudaMemcpyDeviceToHost, stream));
        auto end = std::chrono::system_clock::now();
        std::cout << "inference and gpu postprocess time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;
    }

    CUDA_CHECK(cudaStreamSynchronize(stream));
}

bool parse_args(int argc, char **argv, std::string &wts, std::string &engine, std::string &img_dir, std::string &sub_type, std::string &cuda_post_process) {
    if (argc < 4) return false;
    if (std::string(argv[1]) == "-s" && argc == 5) {
        wts = std::string(argv[2]);
        engine = std::string(argv[3]);
        sub_type = std::string(argv[4]);
    } else if (std::string(argv[1]) == "-d" && argc == 5) {
        engine = std::string(argv[2]);
        img_dir = std::string(argv[3]);
        cuda_post_process = std::string(argv[4]);
    } else {
        return false;
    }
    return true;
}

void calc_distance(std::vector<cv::Mat> &img_batch, std::vector<std::vector<Detection>> &res_batch) {
    int focal_length = 608;
    std::vector<Result_buffer> result_temp_list[3];
    for (size_t i = 0; i < img_batch.size(); i++) {
        if (!i){
            auto &res = res_batch[i];
            cv::Mat img = img_batch[i];
            for (size_t j = 0; j < res.size(); j++) {
                cv::Rect r = get_rect(img, res[j].bbox);
                // real_distance / real_width = focal_distance / pixel_width
                // real_width: 0: 20 cm , 1: 24.5 cm
                int distance = 0;
                if ((int)res[j].class_id == 0){
                    int calc_w = r.width > r.height ? r.width : r.height;
                    distance = 20 * focal_length / calc_w;
                }
                else if((int)res[j].class_id == 1){
                    int calc_w = r.width > r.height ? r.width : r.height;
                    distance = int(24.5 * focal_length / calc_w);
                }
                else if((int)res[j].class_id == 3){
                    int calc_w = r.width > r.height ? r.height : r.width;
                    distance = int(100 * focal_length / calc_w);
                }

                if ((int)res[j].class_id <= 1){
                    result_temp_list[(int)res[j].class_id].emplace_back(
                            (Result_buffer){r.tl().x - r.width/2, r.tl().y - r.height/2, distance});
                }
                else if ((int)res[j].class_id == 3){
                    result_temp_list[(int)res[j].class_id - 1].emplace_back(
                            (Result_buffer){r.tl().x - r.width/2, r.tl().y - r.height/2, distance});
                }

                cv::rectangle(img, r, cv::Scalar(0x27, 0xC1, 0x36), 2);
                cv::putText(img, std::to_string((int) res[j].class_id), cv::Point(r.x, r.y - 1), cv::FONT_HERSHEY_PLAIN,
                            1.2, cv::Scalar(0x00, 0xFF, 0x00), 1);
                cv::putText(img, std::to_string((distance)), cv::Point(r.x, r.y + 15), cv::FONT_HERSHEY_PLAIN,
                            1.2, cv::Scalar(0x00, 0x00, 0xFF), 1);
            }
            detected_src = img;
        }
    }

    // yolov8 label 0:volleyball, 1:basketball, 2:hoop, 3:column
    // serial_com_mode 0:none 
    switch (aim) {
        case 1:{
            if (result_temp_list[0].empty())
                result_buffer[0] = (Result_buffer){0, 0, 0};
            else{
                auto target_iter = result_temp_list[0].begin();
                int min_dist = 10000;
                for (auto iter = result_temp_list[0].begin(); iter != result_temp_list[0].end()-1; iter++){
                    if ((*iter).distance < min_dist){
                        target_iter = iter;
                        min_dist = (*iter).distance;
                    }
                }
                result_buffer[0] = (*target_iter);
            }
            break;
        }
        case 2:{
            if (result_temp_list[1].empty())
                result_buffer[1] = (Result_buffer){0, 0, 0};
            else{
                auto target_iter = result_temp_list[1].begin();
                int min_dist = 10000;
                for (auto iter = result_temp_list[1].begin(); iter != result_temp_list[1].end()-1; iter++){
                    if ((*iter).distance < min_dist){
                        target_iter = iter;
                        min_dist = (*iter).distance;
                    }
                }
                result_buffer[1] = (*target_iter);
            }
            break;
        }
        case 3:{
            if (result_temp_list[2].empty())
                result_buffer[2] = (Result_buffer){0, 0, 0};
            else{
                auto target_iter = result_temp_list[2].begin();
                int min_dist = 10000;
                for (auto iter = result_temp_list[2].begin(); iter != result_temp_list[2].end()-1; iter++){
                    if ((*iter).distance < min_dist){
                        target_iter = iter;
                        min_dist = (*iter).distance;
                    }
                }
                result_buffer[2] = (*target_iter);
            }
            break;
        }
        default:{
            for (auto & i : result_buffer){
                i = (Result_buffer){0, 0, 0};
            }
        }
    }
}

void modeCallback(const std_msgs::UInt8::ConstPtr& msg)
{
    aim = (target)msg->data;
    ROS_INFO("%d", aim);
}

void imageCallback(const sensor_msgs::ImageConstPtr& msg)
{
    cv::Mat src = cv_bridge::toCvShare(msg, "bgr8")->image;
    std::vector<cv::Mat> img_batch;
    img_batch.push_back(src);
    // Preprocess
    cuda_batch_preprocess(img_batch, custom_device_buffers[0], kInputW, kInputH, custom_stream);
    // Run inference
    infer(*custom_context, custom_stream, (void **)custom_device_buffers, custom_output_buffer_host,
          kBatchSize, custom_decode_ptr_host, custom_decode_ptr_device, custom_model_bboxes, custom_cuda_post_process);
    std::vector<std::vector<Detection>> res_batch;
    batch_process(res_batch, custom_decode_ptr_host, img_batch.size(), bbox_element, img_batch);
    // Draw bounding boxes
    calc_distance(img_batch, res_batch);

    // Save detected images
    for (const auto & j : img_batch) {
        if (if_save){
            char base_name[256];   //图像保存
            sprintf(base_name,"/home/hlf/Downloads/myFiles/clac_focal_length/image/%04ld.jpg",count_++);
            cv::imwrite(base_name, j);
        }
        if (if_show){
        cv::imshow("view", j);
        }
    }
}

int main(int argc, char **argv) {
    ros::init(argc, argv, "yolov8");
    ros::NodeHandle nh;

    cudaSetDevice(kGpuId);
    std::string wts_name = "";
    std::string engine_name = "";
    std::string sub_type = "";

    char run_mode = 'd';
    if (run_mode == 's') {
        wts_name = "/home/hlf/catkin_ws/src/yolov8/weights/64061_best.wts";
        engine_name = "/home/hlf/catkin_ws/src/yolov8/engine/64061_best.engine";
        sub_type = "n";
    } else if (run_mode == 'd') {
        engine_name = "/home/hlf/catkin_ws/src/yolov8/engine/64061_best.engine";
        custom_cuda_post_process = "g";
    }


//    if (!parse_args(argc, argv, wts_name, engine_name, img_dir, sub_type, cuda_post_process)) {
//        std::cerr << "Arguments not right!" << std::endl;
//        std::cerr << "./yolov8 -s [.wts] [.engine] [n/s/m/l/x]  // serialize model to plan file" << std::endl;
//        std::cerr << "./yolov8 -d [.engine] ../samples  [c/g]// deserialize plan file and run inference" << std::endl;
//        return -1;
//    }

    // Create a model using the API directly and serialize it to a file
    if (!wts_name.empty()) {
        serialize_engine(wts_name, engine_name, sub_type);
        return 0;
    }

    // Deserialize the engine from file
    deserialize_engine(engine_name, &custom_runtime, &custom_engine, &custom_context);
    CUDA_CHECK(cudaStreamCreate(&custom_stream));
    cuda_preprocess_init(kMaxInputImageSize);
    auto out_dims = custom_engine->getBindingDimensions(1);
    custom_model_bboxes = out_dims.d[0];

    // Prepare cpu and gpu buffers
    prepare_buffer(custom_engine, &custom_device_buffers[0], &custom_device_buffers[1],
                   &custom_output_buffer_host, &custom_decode_ptr_host, &custom_decode_ptr_device, custom_cuda_post_process);

    ros::Subscriber modeSub = nh.subscribe("/detect_mode", 1, &modeCallback);
    image_transport::ImageTransport it(nh);
    image_transport::Subscriber sub = it.subscribe("camera/color/image_raw", 1, &imageCallback);
    image_transport::Publisher detectedImgPub = it.advertise("/detected_img", 1);
    ros::Publisher resultPub = nh.advertise<yolov8::result>("/detect_result", 1);
//    ros::Subscriber paramSub = nh.subscribe("/param", 1, &paramCallback);

    cv::namedWindow("view");
    cv::startWindowThread();

    ros::Rate loop_rate(500);
    while(ros::ok())
    {
        if (if_show){
            //发布检测后的图片
            sensor_msgs::ImagePtr detected_img_msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", detected_src).toImageMsg();
            detectedImgPub.publish(detected_img_msg);
        }

        //发布检测结果
        yolov8::result result_msg;
        if (aim <= 1){
            result_msg.x = result_buffer[aim].x;
            result_msg.y = result_buffer[aim].y;
            result_msg.distance = result_buffer[aim].distance;
            result_msg.direction = 2;
        }
        else{
            result_msg.x = 2;
            result_msg.y = 2;
            result_msg.distance = 2;
            result_msg.distance = 2;
        }

        resultPub.publish(result_msg);
        ros::spinOnce();
        loop_rate.sleep();
    }

    cv::destroyWindow("view");

    // Release stream and buffers
    cudaStreamDestroy(custom_stream);
    CUDA_CHECK(cudaFree(custom_device_buffers[0]));
    CUDA_CHECK(cudaFree(custom_device_buffers[1]));
    CUDA_CHECK(cudaFree(custom_decode_ptr_device));
    delete[] custom_decode_ptr_host;
    delete[] custom_output_buffer_host;
    cuda_preprocess_destroy();
    // Destroy the engine
    delete custom_context;
    delete custom_engine;
    delete custom_runtime;

    return 0;
}

// 用cv的窗口显示检测后的图片
//    cv::namedWindow("view");
//    cv::startWindowThread();
//    cv::destroyWindow("view");
//    if (if_show){
//        cv::imshow("view", img_batch[j]);
//    }

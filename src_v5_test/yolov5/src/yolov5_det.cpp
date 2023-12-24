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

#include <iostream>
#include <chrono>
#include <cmath>

using namespace nvinfer1;

static Logger gLogger;
const static int kOutputSize = kMaxNumOutputBbox * sizeof(Detection) / sizeof(float) + 1;

long int count_=0;

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

float* gpu_buffers[2];
cudaStream_t stream;
IRuntime* runtime = nullptr;
ICudaEngine* engine = nullptr;
IExecutionContext* context = nullptr;
float* cpu_output_buffer = nullptr;

bool parse_args(int argc, char** argv, std::string& wts, std::string& engine, bool& is_p6, float& gd, float& gw, std::string& img_dir) {
    if (argc < 4) return false;
    if (std::string(argv[1]) == "-s" && (argc == 5 || argc == 7)) {
    wts = std::string(argv[2]);
    engine = std::string(argv[3]);
    auto net = std::string(argv[4]);
    if (net[0] == 'n') {
      gd = 0.33;
      gw = 0.25;
    } else if (net[0] == 's') {
      gd = 0.33;
      gw = 0.50;
    } else if (net[0] == 'm') {
      gd = 0.67;
      gw = 0.75;
    } else if (net[0] == 'l') {
      gd = 1.0;
      gw = 1.0;
    } else if (net[0] == 'x') {
      gd = 1.33;
      gw = 1.25;
    } else if (net[0] == 'c' && argc == 7) {
      gd = atof(argv[5]);
      gw = atof(argv[6]);
    } else {
      return false;
    }
    if (net.size() == 2 && net[1] == '6') {
      is_p6 = true;
    }
    } else if (std::string(argv[1]) == "-d" && argc == 4) {
    engine = std::string(argv[2]);
    img_dir = std::string(argv[3]);
    } else {
    return false;
    }
    return true;
}

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
                else if((int)res[j].class_id == 2){
                    int calc_w = r.width > r.height ? r.height : r.width;
                    distance = int(100 * focal_length / calc_w);
                }

                if ((int)res[j].class_id <= 3){
                    result_temp_list[(int)res[j].class_id].emplace_back(
                            (Result_buffer){r.tl().x - r.width/2, r.tl().y - r.height/2, distance});
                }
                cv::rectangle(img, r, cv::Scalar(0x27, 0xC1, 0x36), 2);
                cv::putText(img, std::to_string((int) res[j].class_id), cv::Point(r.x, r.y - 1), cv::FONT_HERSHEY_PLAIN,
                            1.2, cv::Scalar(0x00, 0xFF, 0x00), 1);
                cv::putText(img, std::to_string(distance), cv::Point(r.x, r.y + 15), cv::FONT_HERSHEY_PLAIN,
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
    cuda_batch_preprocess(img_batch, gpu_buffers[0], kInputW, kInputH, stream);

    // Run inference
    infer(*context, stream, (void**)gpu_buffers, cpu_output_buffer, kBatchSize);
//    auto start = std::chrono::system_clock::now();
//    auto end = std::chrono::system_clock::now();
//    std::cout << "inference time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;

    // NMS
    std::vector<std::vector<Detection>> res_batch;
    batch_nms(res_batch, cpu_output_buffer, img_batch.size(), kOutputSize, kConfThresh, kNmsThresh);

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

int main(int argc, char** argv) {
    cudaSetDevice(kGpuId);

    ros::init(argc, argv, "yolov5");
    ros::NodeHandle nh;

    std::string wts_name;
    std::string engine_name;
    bool is_p6 = false;
    float gd = 0.0f, gw = 0.0f;

    char s_or_d = 's';
    if (s_or_d == 's'){
        wts_name = "/home/hlf/ws_repo/v5_ws/src/yolov5/weights/test1.wts";
        engine_name = "/home/hlf/ws_repo/v5_ws/src/yolov5/engine/test1.engine";
        gd = 0.33;
        gw = 0.50;
        is_p6 = false;
    }
    else if (s_or_d == 'd'){
        engine_name = "/home/hlf/ws_repo/v5_ws/src/yolov5/engine/test1.engine";
    }
    //  if (!parse_args(argc, argv, wts_name, engine_name, is_p6, gd, gw, img_dir)) {
    //    std::cerr << "arguments not right!" << std::endl;
    //    std::cerr << "./yolov5_det -s [.wts] [.engine] [n/s/m/l/x/n6/s6/m6/l6/x6 or c/c6 gd gw]  // serialize model to plan file" << std::endl;
    //    std::cerr << "./yolov5_det -d [.engine] ../images  // deserialize plan file and run inference" << std::endl;
    //    return -1;
    //  }


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

    //  // Read images from directory
    //  std::vector<std::string> file_names;
    //  if (read_files_in_dir(img_dir.c_str(), file_names) < 0) {
    //    std::cerr << "read_files_in_dir failed." << std::endl;
    //    return -1;
    //  }

    ros::Subscriber modeSub = nh.subscribe("/detect_mode", 1, &modeCallback);
    image_transport::ImageTransport it(nh);
    image_transport::Subscriber sub = it.subscribe("camera/color/image_raw", 1, &imageCallback);
    image_transport::Publisher detectedImgPub = it.advertise("/detected_img", 1);
    ros::Publisher resultPub = nh.advertise<yolov5::result>("/detect_result", 1);
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
        yolov5::result result_msg;
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

    //  // batch predict
    //  for (size_t i = 0; i < file_names.size(); i += kBatchSize) {
    //    // Get a batch of images
    //    std::vector<cv::Mat> img_batch;
    //    std::vector<std::string> img_name_batch;
    //    for (size_t j = i; j < i + kBatchSize && j < file_names.size(); j++) {
    //      cv::Mat img = cv::imread(img_dir + "/" + file_names[j]);
    //      img_batch.push_back(img);
    //      img_name_batch.push_back(file_names[j]);
    //    }
    //
    //    // Preprocess
    //    cuda_batch_preprocess(img_batch, gpu_buffers[0], kInputW, kInputH, stream);
    //
    //    // Run inference
    //    auto start = std::chrono::system_clock::now();
    //    infer(*context, stream, (void**)gpu_buffers, cpu_output_buffer, kBatchSize);
    //    auto end = std::chrono::system_clock::now();
    //    std::cout << "inference time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;
    //
    //    // NMS
    //    std::vector<std::vector<Detection>> res_batch;
    //    batch_nms(res_batch, cpu_output_buffer, img_batch.size(), kOutputSize, kConfThresh, kNmsThresh);
    //
    //    // Draw bounding boxes
    //    draw_bbox(img_batch, res_batch);
    //
    //    // Save images
    //    for (size_t j = 0; j < img_batch.size(); j++) {
    //      cv::imwrite("_" + img_name_batch[j], img_batch[j]);
    //    }
    //  }

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


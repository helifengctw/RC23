#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <opencv2/highgui/highgui.hpp>
#include <cv_bridge/cv_bridge.h>
#include <usb_camera_pkg/realsense_dev.h>

long int count_=0;
float depth_value = 0;
usb_camera_pkg::realsense_dev data_to_pub;   //待发布数据


//依赖包有std_msgs（消息传递），roscpp（c++），cv_bridge（ros和opencv图像消息转换），sensor_msgs（传感器消息），image_transport（图像编码传输）
void imageCallback(const sensor_msgs::ImageConstPtr& msg)
{
    try
    {
        auto cv_ptr =  cv_bridge::toCvCopy(msg, "bgr8");
        cv::Mat img = cv_bridge::toCvShare(msg, "bgr8")->image;
        cv::imshow("BGR", img);

        char base_name[256];   //图像保存
        sprintf(base_name,"/home/hlf/Downloads/myFiles/yolov8_init/database/%04ld.jpg",count_++);
        cv::imwrite(base_name,cv_ptr->image);
    }
    catch (cv_bridge::Exception& e)
    {
        ROS_ERROR("Could not convert from '%s' to 'bgr8'.", msg->encoding.c_str());
    }
}


int main(int argc, char **argv)
{
    ros::init(argc, argv, "img_viewer");
    ros::NodeHandle nh;
    cv::namedWindow("BGR");
    cv::startWindowThread();
    image_transport::ImageTransport it(nh);
    image_transport::Subscriber image_sub = it.subscribe("camera/color/image_raw", 1, imageCallback);
    ros::spin();
    cv::destroyWindow("BGR");
    return 0;
}

//void depthCallback(const sensor_msgs::ImageConstPtr& msg)
//{
//    auto img_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::TYPE_16UC1);
//    cv::Mat depth_img = img_ptr -> image;
//    cv::imshow("depth", depth_img);
//    ushort d = depth_img.at<ushort>(depth_img.rows/2, depth_img.cols/2);
//    depth_value = float(d) / 1000;
//    std::cout << depth_value << std::endl;
//}

//ros::AsyncSpinner spinner(2);
//spinner.start();
//cv::namedWindow("depth");
//image_transport::Subscriber depth_sub = it.subscribe("/camera/aligned_depth_to_color/image_raw",1,depthCallback);   //订阅深度图像
////ros::Subscriber element_sub = nh.subscribe<sensor_msgs::Image>("/camera/aligned_depth_to_color/image_raw",100,pixelCallback);     //订阅像素点坐标
//ros::Publisher depth_pub = nh.advertise<usb_camera_pkg::realsense_dev>("/depth_info", 10);
//ros::waitForShutdown();
//cv::destroyWindow("depth");
//data_to_pub.depth_value = 0;    //初始化深度值
//ros::Rate rate(20.0);    //设定自循环的频率
//while(ros::ok)
//{
////data_to_pub.header.stamp = ros::Time::now();
//data_to_pub.depth_value = depth_value;     //depth_pic.rows/2,depth_pic.cols/2  为像素点
//depth_pub.publish(data_to_pub);
//}
//ros::Duration(10).sleep();    //设定自循环的频率
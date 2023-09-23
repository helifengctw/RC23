#include <ros/ros.h>
#include <serial/serial.h>
#include <iostream>
#include <std_msgs/UInt8.h>
#include <image_transport/image_transport.h>
#include <opencv2/highgui/highgui.hpp>
#include <cv_bridge/cv_bridge.h>
#include "result.h"


serial::Serial sp;  //创建一个serial类

void resultCallback(const yolov8::result ::ConstPtr& msg) {
    uint8_t send[8] = {1, 2, 3, 4, 5, 6, 7 , 8};

    send[0] = (uint8_t) msg->x;
    send[1] = (uint8_t) ((msg->x) >> 8);
    send[2] = (uint8_t) msg->y;
    send[3] = (uint8_t) ((msg->y) >> 8);
    send[4] = (uint8_t) msg->distance;
    send[5] = (uint8_t) ((msg->distance) >> 8);
    send[6] = (uint8_t) (msg->direction + 180);
    send[7] = (uint8_t) ((msg->direction + 180) >> 8);

    sp.write(send, 8);  //把数据发送回去
    std::cout << "send: " << std::hex << (send[0] & 0xff) << std::endl;
}

void imageCallback(const sensor_msgs::ImageConstPtr& msg){
    cv::Mat src = cv_bridge::toCvShare(msg, "bgr8")->image;
    cv::imshow("BGR", src);
}

int main(int argc, char** argv)
{
    ros::init(argc, argv, "serial_com");
    ros::NodeHandle nh;
    ros::Publisher modePub = nh.advertise<std_msgs::UInt8>("/detect_mode", 1);
    ros::Subscriber resultSub = nh.subscribe("/detect_result", 1, &resultCallback);

    serial::Timeout to = serial::Timeout::simpleTimeout(1000); //创建timeout
    sp.setPort("/dev/ttyACM0"); //设置要打开的串口名称
    sp.setBaudrate(115200); //设置串口通信的波特率
    sp.setTimeout(to); //串口设置timeout

    try{
        sp.open(); //打开串口
    } catch(serial::IOException& e){
        ROS_ERROR_STREAM("Unable to open port.");
        return -1;
    }
    //判断串口是否打开成功
    if (sp.isOpen()) {
        ROS_INFO_STREAM("/dev/ttyACM0 is opened.");
    } else {
        return -1;
    }

    ros::Rate loop_rate(500);
    while(ros::ok())
    {
        size_t n = sp.available();  //获取缓冲区内的字节数

        if(n!=0)
        {
            uint8_t buffer[1024];
            n = sp.read(buffer, n);  //读出数据

            std::cout << "recieve: " << std::hex << (buffer[0] & 0xff) << " , n = " << n << std::endl;

            std_msgs::UInt8 mode_msgs;
            mode_msgs.data = buffer[0];
            modePub.publish(mode_msgs);

        }
        loop_rate.sleep();
    }

    sp.close();  //关闭串口

    return 0;
}

//int main(int argc, char** argv) {
//    ros::init(argc, argv, "serial_port");
//    //创建句柄（虽然后面没用到这个句柄，但如果不创建，运行时进程会出错）
//    ros::NodeHandle n;
//
//    //创建一个serial类
//    serial::Serial sp;
//    //创建timeout
//    serial::Timeout to = serial::Timeout::simpleTimeout(100);
//    //设置要打开的串口名称
//    sp.setPort("/dev/ttyACM0");
//    //设置串口通信的波特率
//    sp.setBaudrate(115200);
//    //串口设置timeout
//    sp.setTimeout(to);
//
//    try {
//        //打开串口
//        sp.open();
//    }
//    catch (serial::IOException &e) {
//        ROS_ERROR_STREAM("Unable to open port.");
//        return -1;
//    }
//
//    //判断串口是否打开成功
//    if (sp.isOpen()) {
//        ROS_INFO_STREAM("/dev/ttyUSB0 is opened.");
//    } else {
//        return -1;
//    }
//
//    ros::Rate loop_rate(500);
//    while (ros::ok()) {
//        //获取缓冲区内的字节数
//        size_t n = sp.available();
//        if (n != 0) {
//            uint8_t buffer[1024];
//            //读出数据
//            n = sp.read(buffer, n);
//
//            std::cout << std::hex << (buffer[0] & 0xff) << " , n = " << n <<  std::endl;
////            for (int i = 0; i < n; i++) {
////                //16进制的方式打印到屏幕
////                std::cout << std::hex << (buffer[i] & 0xff) << " ";
////            }
//            //把数据发送回去
//            uint8_t send[8] = {1, 2, 3, 0xb4, 5, 6, 7 , 8};
//            sp.write(send, 8);
//            std::cout << "send: " << std::hex << (send[0] & 0xff) << std::endl;
//        }
//        loop_rate.sleep();
//    }
//
//    //关闭串口
//    sp.close();
//
//    return 0;
//}

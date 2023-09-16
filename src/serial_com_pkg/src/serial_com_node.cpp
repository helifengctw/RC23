#include <ros/ros.h>
#include <serial/serial.h>
#include <iostream>
#include <std_msgs/UInt8.h>
#include <image_transport/image_transport.h>
#include <opencv2/highgui/highgui.hpp>
#include <cv_bridge/cv_bridge.h>
#include "mode.h"
#include "result.h"

uint8_t send[3];

void resultCallback(const yolov8::result ::ConstPtr& msg){
    send[0] = (uint8_t)msg->x;
    send[1] = (uint8_t)msg->y;
    send[2] = (uint8_t)msg->distance;
//    send[0] = (uint8_t)msg->x;
//    send[1] = (uint8_t)((msg->x) >> 8);
//    send[2] = (uint8_t)msg->y;
//    send[3] = (uint8_t)((msg->y) >> 8);
//    send[4] = (uint8_t)msg->distance;
//    send[5] = (uint8_t)((msg->distance) >> 8);
//    send[6] = (uint8_t)(msg->direction + 180);
//    send[7] = (uint8_t)((msg->direction + 180) >> 8);

//    unsigned short* test = (unsigned short*)send;
//    sp.write(send, 8);
    //for(int i = 0; i < 8; i++)
    //{
    //std::cout << std::hex << (char)send[i] << std::endl;
    //}
}

void imageCallback(const sensor_msgs::ImageConstPtr& msg){
    cv::Mat src = cv_bridge::toCvShare(msg, "bgr8")->image;
    cv::imshow("BGR", src);
}


int main(int argc, char** argv)
{
    ros::init(argc, argv, "serial_com");
    ros::NodeHandle nh;
    image_transport::ImageTransport it(nh);
    image_transport::Subscriber sub = it.subscribe("/detected_img", 1, &imageCallback);
    ros::Publisher modePub = nh.advertise<serial_com_pkg::mode>("/detect_mode", 1);
    ros::Subscriber resultSub = nh.subscribe("/detect_result", 1, &resultCallback);
    std_msgs::UInt8 mode_msgs;

    cv::namedWindow("detected_img");
    cv::startWindowThread();

    uint8_t receive[3] = {2, 2, 2};

    serial::Serial sp;  //创建一个serial对象
    serial::Timeout to = serial::Timeout::simpleTimeout(100);  //创建timeout
    sp.setPort("/dev/ttyACM0");  //设置要打开的串口名称
    sp.setBaudrate(115200);  //设置串口通信的波特率
    sp.setTimeout(to);  //串口设置timeout

    try {
        sp.open();  //打开串口
    }
    catch(serial::IOException& e) {
        ROS_ERROR_STREAM("Unable to open port.");
        return -1;
    }

    //判断串口是否打开成功
    if(sp.isOpen()) {
        ROS_INFO_STREAM("/dev/ttyACM0 is opened.");
    }
    else {
        return -1;
    }

    ros::Rate loop_rate(500);
    while(ros::ok())
    {
        //获取缓冲区内的字节数
        size_t n = sp.available();
        if(n!=0)
        {
            n = sp.read(receive, 3);  //读出数据

            for(int i=0; i<n; i++){
                mode_msgs.data = receive[0];
                modePub.publish(mode_msgs);
                ROS_INFO("mode: %d", receive[0]);
//                std::cout << std::hex << (buffer[i] & 0xff) << " ";  //16进制的方式打印到屏幕
            }
        }
        sp.write(send, 3);
        ros::spinOnce();
        loop_rate.sleep();
    }

    sp.close();  //关闭串口
    cv::destroyWindow("detected_img");

    return 0;
}


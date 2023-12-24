#!bin/bash

source ~/.bashrc
source ~/catkin_ws/devel/setup.bash

gnome-terminal -x bash -c "realsense-viewer & ; sleep 2; exit"
echo "realsense-viewer has been launched, d435i initialized !!"
sleep 2

gnome-terminal -x bash -c "realsense-viewer & ; sleep 2; exit"
echo "realsense-viewer has been launched, d435i initialized again !!"
sleep 2

gnome-terminal -x bash -c "roscore;"
echo "successfully run roscore !!"
sleep 1
gnome-terminal -x bash -c "roslaunch realsense2_camera rs_camera.launch;"
echo "successfuuly launch rs_camera !!"
sleep 2
gnome-terminal -x bash -c "rosrun yolov8 yolov8;"
echo "successfuuly run yolov8 !!"
sleep 2
# gnome-terminal -x bash -c "rosrun serial_com_pkg serial_com"
# echo "successfuuly run serial_com !!"




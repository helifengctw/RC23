# generated from catkin/cmake/template/pkg.context.pc.in
CATKIN_PACKAGE_PREFIX = ""
PROJECT_PKG_CONFIG_INCLUDE_DIRS = "${prefix}/include".split(';') if "${prefix}/include" != "" else []
PROJECT_CATKIN_DEPENDS = "message_runtime;roscpp;rospy;serial;std_msgs".replace(';', ' ')
PKG_CONFIG_LIBRARIES_WITH_PREFIX = "-lserial_com_pkg".split(';') if "-lserial_com_pkg" != "" else []
PROJECT_NAME = "serial_com_pkg"
PROJECT_SPACE_DIR = "/usr/local"
PROJECT_VERSION = "0.0.0"

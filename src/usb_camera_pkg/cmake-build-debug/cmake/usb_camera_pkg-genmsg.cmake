# generated from genmsg/cmake/pkg-genmsg.cmake.em

message(STATUS "usb_camera_pkg: 1 messages, 0 services")

set(MSG_I_FLAGS "-Iusb_camera_pkg:/home/hlf/catkin_ws/src/usb_camera_pkg/msg;-Istd_msgs:/opt/ros/noetic/share/std_msgs/cmake/../msg")

# Find all generators
find_package(gencpp REQUIRED)
find_package(geneus REQUIRED)
find_package(genlisp REQUIRED)
find_package(gennodejs REQUIRED)
find_package(genpy REQUIRED)

add_custom_target(usb_camera_pkg_generate_messages ALL)

# verify that message/service dependencies have not changed since configure



get_filename_component(_filename "/home/hlf/catkin_ws/src/usb_camera_pkg/msg/realsense_dev.msg" NAME_WE)
add_custom_target(_usb_camera_pkg_generate_messages_check_deps_${_filename}
  COMMAND ${CATKIN_ENV} ${PYTHON_EXECUTABLE} ${GENMSG_CHECK_DEPS_SCRIPT} "usb_camera_pkg" "/home/hlf/catkin_ws/src/usb_camera_pkg/msg/realsense_dev.msg" ""
)

#
#  langs = gencpp;geneus;genlisp;gennodejs;genpy
#

### Section generating for lang: gencpp
### Generating Messages
_generate_msg_cpp(usb_camera_pkg
  "/home/hlf/catkin_ws/src/usb_camera_pkg/msg/realsense_dev.msg"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/usb_camera_pkg
)

### Generating Services

### Generating Module File
_generate_module_cpp(usb_camera_pkg
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/usb_camera_pkg
  "${ALL_GEN_OUTPUT_FILES_cpp}"
)

add_custom_target(usb_camera_pkg_generate_messages_cpp
  DEPENDS ${ALL_GEN_OUTPUT_FILES_cpp}
)
add_dependencies(usb_camera_pkg_generate_messages usb_camera_pkg_generate_messages_cpp)

# add dependencies to all check dependencies targets
get_filename_component(_filename "/home/hlf/catkin_ws/src/usb_camera_pkg/msg/realsense_dev.msg" NAME_WE)
add_dependencies(usb_camera_pkg_generate_messages_cpp _usb_camera_pkg_generate_messages_check_deps_${_filename})

# target for backward compatibility
add_custom_target(usb_camera_pkg_gencpp)
add_dependencies(usb_camera_pkg_gencpp usb_camera_pkg_generate_messages_cpp)

# register target for catkin_package(EXPORTED_TARGETS)
list(APPEND ${PROJECT_NAME}_EXPORTED_TARGETS usb_camera_pkg_generate_messages_cpp)

### Section generating for lang: geneus
### Generating Messages
_generate_msg_eus(usb_camera_pkg
  "/home/hlf/catkin_ws/src/usb_camera_pkg/msg/realsense_dev.msg"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/usb_camera_pkg
)

### Generating Services

### Generating Module File
_generate_module_eus(usb_camera_pkg
  ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/usb_camera_pkg
  "${ALL_GEN_OUTPUT_FILES_eus}"
)

add_custom_target(usb_camera_pkg_generate_messages_eus
  DEPENDS ${ALL_GEN_OUTPUT_FILES_eus}
)
add_dependencies(usb_camera_pkg_generate_messages usb_camera_pkg_generate_messages_eus)

# add dependencies to all check dependencies targets
get_filename_component(_filename "/home/hlf/catkin_ws/src/usb_camera_pkg/msg/realsense_dev.msg" NAME_WE)
add_dependencies(usb_camera_pkg_generate_messages_eus _usb_camera_pkg_generate_messages_check_deps_${_filename})

# target for backward compatibility
add_custom_target(usb_camera_pkg_geneus)
add_dependencies(usb_camera_pkg_geneus usb_camera_pkg_generate_messages_eus)

# register target for catkin_package(EXPORTED_TARGETS)
list(APPEND ${PROJECT_NAME}_EXPORTED_TARGETS usb_camera_pkg_generate_messages_eus)

### Section generating for lang: genlisp
### Generating Messages
_generate_msg_lisp(usb_camera_pkg
  "/home/hlf/catkin_ws/src/usb_camera_pkg/msg/realsense_dev.msg"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/usb_camera_pkg
)

### Generating Services

### Generating Module File
_generate_module_lisp(usb_camera_pkg
  ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/usb_camera_pkg
  "${ALL_GEN_OUTPUT_FILES_lisp}"
)

add_custom_target(usb_camera_pkg_generate_messages_lisp
  DEPENDS ${ALL_GEN_OUTPUT_FILES_lisp}
)
add_dependencies(usb_camera_pkg_generate_messages usb_camera_pkg_generate_messages_lisp)

# add dependencies to all check dependencies targets
get_filename_component(_filename "/home/hlf/catkin_ws/src/usb_camera_pkg/msg/realsense_dev.msg" NAME_WE)
add_dependencies(usb_camera_pkg_generate_messages_lisp _usb_camera_pkg_generate_messages_check_deps_${_filename})

# target for backward compatibility
add_custom_target(usb_camera_pkg_genlisp)
add_dependencies(usb_camera_pkg_genlisp usb_camera_pkg_generate_messages_lisp)

# register target for catkin_package(EXPORTED_TARGETS)
list(APPEND ${PROJECT_NAME}_EXPORTED_TARGETS usb_camera_pkg_generate_messages_lisp)

### Section generating for lang: gennodejs
### Generating Messages
_generate_msg_nodejs(usb_camera_pkg
  "/home/hlf/catkin_ws/src/usb_camera_pkg/msg/realsense_dev.msg"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/usb_camera_pkg
)

### Generating Services

### Generating Module File
_generate_module_nodejs(usb_camera_pkg
  ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/usb_camera_pkg
  "${ALL_GEN_OUTPUT_FILES_nodejs}"
)

add_custom_target(usb_camera_pkg_generate_messages_nodejs
  DEPENDS ${ALL_GEN_OUTPUT_FILES_nodejs}
)
add_dependencies(usb_camera_pkg_generate_messages usb_camera_pkg_generate_messages_nodejs)

# add dependencies to all check dependencies targets
get_filename_component(_filename "/home/hlf/catkin_ws/src/usb_camera_pkg/msg/realsense_dev.msg" NAME_WE)
add_dependencies(usb_camera_pkg_generate_messages_nodejs _usb_camera_pkg_generate_messages_check_deps_${_filename})

# target for backward compatibility
add_custom_target(usb_camera_pkg_gennodejs)
add_dependencies(usb_camera_pkg_gennodejs usb_camera_pkg_generate_messages_nodejs)

# register target for catkin_package(EXPORTED_TARGETS)
list(APPEND ${PROJECT_NAME}_EXPORTED_TARGETS usb_camera_pkg_generate_messages_nodejs)

### Section generating for lang: genpy
### Generating Messages
_generate_msg_py(usb_camera_pkg
  "/home/hlf/catkin_ws/src/usb_camera_pkg/msg/realsense_dev.msg"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/usb_camera_pkg
)

### Generating Services

### Generating Module File
_generate_module_py(usb_camera_pkg
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/usb_camera_pkg
  "${ALL_GEN_OUTPUT_FILES_py}"
)

add_custom_target(usb_camera_pkg_generate_messages_py
  DEPENDS ${ALL_GEN_OUTPUT_FILES_py}
)
add_dependencies(usb_camera_pkg_generate_messages usb_camera_pkg_generate_messages_py)

# add dependencies to all check dependencies targets
get_filename_component(_filename "/home/hlf/catkin_ws/src/usb_camera_pkg/msg/realsense_dev.msg" NAME_WE)
add_dependencies(usb_camera_pkg_generate_messages_py _usb_camera_pkg_generate_messages_check_deps_${_filename})

# target for backward compatibility
add_custom_target(usb_camera_pkg_genpy)
add_dependencies(usb_camera_pkg_genpy usb_camera_pkg_generate_messages_py)

# register target for catkin_package(EXPORTED_TARGETS)
list(APPEND ${PROJECT_NAME}_EXPORTED_TARGETS usb_camera_pkg_generate_messages_py)



if(gencpp_INSTALL_DIR AND EXISTS ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/usb_camera_pkg)
  # install generated code
  install(
    DIRECTORY ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/usb_camera_pkg
    DESTINATION ${gencpp_INSTALL_DIR}
  )
endif()
if(TARGET std_msgs_generate_messages_cpp)
  add_dependencies(usb_camera_pkg_generate_messages_cpp std_msgs_generate_messages_cpp)
endif()

if(geneus_INSTALL_DIR AND EXISTS ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/usb_camera_pkg)
  # install generated code
  install(
    DIRECTORY ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/usb_camera_pkg
    DESTINATION ${geneus_INSTALL_DIR}
  )
endif()
if(TARGET std_msgs_generate_messages_eus)
  add_dependencies(usb_camera_pkg_generate_messages_eus std_msgs_generate_messages_eus)
endif()

if(genlisp_INSTALL_DIR AND EXISTS ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/usb_camera_pkg)
  # install generated code
  install(
    DIRECTORY ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/usb_camera_pkg
    DESTINATION ${genlisp_INSTALL_DIR}
  )
endif()
if(TARGET std_msgs_generate_messages_lisp)
  add_dependencies(usb_camera_pkg_generate_messages_lisp std_msgs_generate_messages_lisp)
endif()

if(gennodejs_INSTALL_DIR AND EXISTS ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/usb_camera_pkg)
  # install generated code
  install(
    DIRECTORY ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/usb_camera_pkg
    DESTINATION ${gennodejs_INSTALL_DIR}
  )
endif()
if(TARGET std_msgs_generate_messages_nodejs)
  add_dependencies(usb_camera_pkg_generate_messages_nodejs std_msgs_generate_messages_nodejs)
endif()

if(genpy_INSTALL_DIR AND EXISTS ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/usb_camera_pkg)
  install(CODE "execute_process(COMMAND \"/usr/bin/python3\" -m compileall \"${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/usb_camera_pkg\")")
  # install generated code
  install(
    DIRECTORY ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/usb_camera_pkg
    DESTINATION ${genpy_INSTALL_DIR}
  )
endif()
if(TARGET std_msgs_generate_messages_py)
  add_dependencies(usb_camera_pkg_generate_messages_py std_msgs_generate_messages_py)
endif()

cmake_minimum_required(VERSION 3.15)
project(higgsr_interface)

# 개발자용(dev) 경고를 모두 숨깁니다.
set(CMAKE_SUPPRESS_DEVELOPER_WARNINGS TRUE)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(builtin_interfaces REQUIRED)

set(msg_files
  "msg/PointCloudInfo.msg"
  "msg/Keypoints.msg"
  "msg/DensityMap.msg"
)

set(srv_files
  "srv/SetGlobalMap.srv"
  "srv/RegisterScan.srv"
  "srv/ProcessFiles.srv"
)

rosidl_generate_interfaces(${PROJECT_NAME}
  ${msg_files}
  ${srv_files}
  DEPENDENCIES sensor_msgs geometry_msgs nav_msgs std_msgs builtin_interfaces
)
set(CMAKE_WARN_DEPRECATED OFF)


if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

# ament 패키지 완료
ament_package()
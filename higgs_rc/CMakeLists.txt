cmake_minimum_required(VERSION 3.18)
project(higgs_rc)

# Default to C++17
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic -O3 -march=native)
endif()

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_components REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_eigen REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(higgsr_interface REQUIRED)

# External libraries
find_package(PCL REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(OpenMP REQUIRED)
find_package(nlohmann_json 3.2.0 QUIET)

# Include directories
include_directories(
  include
  ${PCL_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
)

# Core library
add_library(higgs_rc_core SHARED
  src/core/feature_extraction.cpp
  src/core/registration.cpp
  src/core/utils.cpp
)

target_link_libraries(higgs_rc_core
  ${PCL_LIBRARIES}
  OpenMP::OpenMP_CXX
)

ament_target_dependencies(higgs_rc_core
  rclcpp
  sensor_msgs
  geometry_msgs
  higgsr_interface
)

# HiGGSR Server Node
add_executable(higgsr_server_node
  src/nodes/higgsr_server_node.cpp
  src/nodes/higgsr_server_main.cpp
)

target_link_libraries(higgsr_server_node
  higgs_rc_core
)

if(nlohmann_json_FOUND)
  target_link_libraries(higgsr_server_node nlohmann_json::nlohmann_json)
endif()

ament_target_dependencies(higgsr_server_node
  rclcpp
  sensor_msgs
  geometry_msgs
  visualization_msgs
  tf2_ros
  tf2_eigen
  higgsr_interface
)

# LiDAR Client Node - TODO: Implement
# add_executable(lidar_client_node
#   src/nodes/lidar_client_node.cpp
# )
# 
# target_link_libraries(lidar_client_node
#   higgs_rc_core
# )
# 
# ament_target_dependencies(lidar_client_node
#   rclcpp
#   sensor_msgs
#   geometry_msgs
#   visualization_msgs
#   tf2_ros
#   tf2_geometry_msgs
#   higgsr_interface
# )

# File Processor Node - TODO: Implement
# add_executable(file_processor_node
#   src/nodes/file_processor_node.cpp
# )
# 
# target_link_libraries(file_processor_node
#   higgs_rc_core
# )
# 
# ament_target_dependencies(file_processor_node
#   rclcpp
#   sensor_msgs
#   geometry_msgs
#   visualization_msgs
#   higgsr_interface
# )

# Install executables
install(TARGETS
  higgsr_server_node
  # lidar_client_node  # TODO
  # file_processor_node  # TODO
  DESTINATION lib/${PROJECT_NAME}
)

# Install libraries
install(TARGETS
  higgs_rc_core
  EXPORT higgs_rc_targets
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  RUNTIME DESTINATION bin
)

# Install header files
install(DIRECTORY include/
  DESTINATION include
)

# Install Python scripts for visualization
install(PROGRAMS
  scripts/visualization.py
  scripts/visualization_node.py
  scripts/test_higgsr_system.py
  scripts/test_rviz_visualization.py
  DESTINATION lib/${PROJECT_NAME}
)

# Install config files
install(DIRECTORY config/
  DESTINATION share/${PROJECT_NAME}/config
)

# Install launch files
install(DIRECTORY launch/
  DESTINATION share/${PROJECT_NAME}/launch
)

# Export dependencies
ament_export_targets(higgs_rc_targets HAS_LIBRARY_TARGET)
ament_export_dependencies(
  rclcpp
  sensor_msgs
  geometry_msgs
  nav_msgs
  std_msgs
  visualization_msgs
  tf2_ros
  tf2_eigen
  higgsr_interface
  PCL
  Eigen3
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  find_package(ament_cmake_gtest REQUIRED)
  
  # Linting
  ament_lint_auto_find_test_dependencies()
  
  # Unit tests - TODO: Implement
  # ament_add_gtest(test_feature_extraction test/test_feature_extraction.cpp)
  # target_link_libraries(test_feature_extraction higgs_rc_core)
  # 
  # ament_add_gtest(test_registration test/test_registration.cpp)
  # target_link_libraries(test_registration higgs_rc_core)
endif()

ament_package()
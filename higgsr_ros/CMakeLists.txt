cmake_minimum_required(VERSION 3.16)
project(higgsr_ros)

# C++17 표준 사용
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ===================================================================
# 1. ROS 및 기타 의존성 패키지 찾기
# ===================================================================

# pybind11, PCL 등 외부 패키지에서 발생하는 모든 개발자용(dev) 경고를 숨깁니다.
set(CMAKE_SUPPRESS_DEVELOPER_WARNINGS TRUE)

find_package(ament_cmake_python REQUIRED)
find_package(pybind11 REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(PCL REQUIRED)

# 🚀 OpenMP for 멀티스레드 C++ 가속화
find_package(OpenMP REQUIRED)

# Python 실행파일 찾기 (버전 감지용)
find_package(Python3 REQUIRED COMPONENTS Interpreter)

# ROS 2 Python 의존성
find_package(rclpy REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_py REQUIRED)
find_package(higgsr_interface REQUIRED)

# Python 패키지 설치
ament_python_install_package(${PROJECT_NAME})

# Python executable 스크립트들 설치
install(PROGRAMS
    higgsr_ros/nodes/higgsr_server_node.py
    higgsr_ros/nodes/lidar_client_node.py
    higgsr_ros/nodes/file_processor_node.py
    higgsr_ros/nodes/file_processor_client_node.py
    higgsr_ros/visualization/visualization_node.py
    higgsr_ros/scripts/test_higgsr_system.py
    higgsr_ros/scripts/test_rviz_visualization.py
    DESTINATION lib/${PROJECT_NAME}
)

# ==========================================================
# 4. C++ 확장 모듈 빌드
# ==========================================================
set(CPP_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/higgsr_ros/core/cpp_src)
if(EXISTS ${CPP_SRC_DIR}/bindings.cpp)
    message(STATUS "C++ source files found, building extension module...")

    pybind11_add_module(higgsr_core_cpp
        ${CPP_SRC_DIR}/bindings.cpp
        ${CPP_SRC_DIR}/feature_extraction.cpp
        ${CPP_SRC_DIR}/registration.cpp
    )

    target_include_directories(higgsr_core_cpp PUBLIC
        $<BUILD_INTERFACE:${CPP_SRC_DIR}/include>
        $<INSTALL_INTERFACE:include>
        ${Eigen3_INCLUDE_DIRS}
        ${PCL_INCLUDE_DIRS}
    )

    target_link_libraries(higgsr_core_cpp PUBLIC
        Eigen3::Eigen
        ${PCL_LIBRARIES}
        OpenMP::OpenMP_CXX
    )

    # Python 버전에 따른 동적 경로 설정
    execute_process(
        COMMAND ${Python3_EXECUTABLE} -c "import sysconfig; print(sysconfig.get_python_version())"
        OUTPUT_VARIABLE PYTHON_VERSION
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    
    install(TARGETS higgsr_core_cpp
        DESTINATION "lib/python${PYTHON_VERSION}/site-packages/${PROJECT_NAME}/core"
    )

    set(CPP_MODULE_BUILT TRUE)
    message(STATUS "C++ extension module configuration complete")
else()
    set(CPP_MODULE_BUILT FALSE)
    message(STATUS "C++ source files not found, Python-only build")
endif()

# ===================================================================
# 5. 기타 자원 및 설정 파일 설치 (수정 없음)
# ===================================================================
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/config)
    install(DIRECTORY config/ DESTINATION share/${PROJECT_NAME}/config)
endif()
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/resource)
    install(DIRECTORY resource/ DESTINATION share/ament_index/resource_index/packages)
endif()
install(FILES package.xml DESTINATION share/${PROJECT_NAME})

# 테스트 설정
if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

# ament 패키지 완료
ament_package()

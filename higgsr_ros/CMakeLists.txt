cmake_minimum_required(VERSION 3.16)
project(higgsr_ros)

# ament_cmake_python 사용 (Python 패키지 + 향후 C++ 확장 지원)
find_package(ament_cmake_python REQUIRED)

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

# 설정 파일들 설치 (있는 경우)
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/config)
    install(DIRECTORY config/
        DESTINATION share/${PROJECT_NAME}/config
        FILES_MATCHING PATTERN "*.yaml" PATTERN "*.rviz"
    )
endif()

# 리소스 파일 설치 (있는 경우)
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/resource)
    install(DIRECTORY resource/
        DESTINATION share/ament_index/resource_index/packages
        FILES_MATCHING PATTERN "${PROJECT_NAME}"
    )
endif()

# Package.xml 설치
install(FILES package.xml
    DESTINATION share/${PROJECT_NAME}
)

# 빌드 정보 출력
message(STATUS "=== HiGGSR ROS Build Configuration ===")
message(STATUS "Project: ${PROJECT_NAME}")
message(STATUS "Build type: Python-only (ament_cmake_python)")
message(STATUS "C++ extensions: Disabled (framework ready)")
message(STATUS "Implementation: Python with Numba JIT acceleration")
message(STATUS "=====================================")

# 테스트 설정
if(BUILD_TESTING)
    find_package(ament_lint_auto REQUIRED)
    ament_lint_auto_find_test_dependencies()
endif()

# ament 패키지 완료
ament_package() 
cmake_minimum_required(VERSION 3.10)

# CMake Policy Settings to suppress warnings
cmake_policy(SET CMP0144 NEW) # find_package uses upper-case <PACKAGENAME>_ROOT variables
cmake_policy(SET CMP0167 NEW) # FindBoost module removed
cmake_policy(SET CMP0148 NEW) # FindPythonInterp and FindPythonLibs modules removed

project(HiGGSR VERSION 0.1.0)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 최적화 옵션 설정
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# 외부 라이브러리 의존성
find_package(Eigen3 REQUIRED)
find_package(PCL REQUIRED)
find_package(OpenMP)
# find_package(PythonLibs REQUIRED) # 이전 방식
find_package(Python COMPONENTS Interpreter Development REQUIRED) # 새 방식
find_package(pybind11 CONFIG REQUIRED)

# 컴파일러 플래그
if(OPENMP_FOUND)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

# 헤더 파일 경로 설정
include_directories(
  ${PROJECT_SOURCE_DIR}/include
  ${PCL_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
  ${Python_INCLUDE_DIRS} # Python 헤더 경로 추가
  ${pybind11_INCLUDE_DIRS} # pybind11 헤더 경로 추가 (필요시)
)

# 소스 파일 추가
file(GLOB_RECURSE SOURCES "src/*.cpp")

# Python 모듈 생성 (pybind11 사용)
pybind11_add_module(${PROJECT_NAME}_cpp 
    ${SOURCES}
)

# Python 모듈에 링크
target_link_libraries(${PROJECT_NAME}_cpp PRIVATE
  ${PCL_LIBRARIES}
  Eigen3::Eigen
  pybind11::module # pybind11이 Python 라이브러리를 연결할 것임
  # ${PYTHON_LIBRARIES} # 제거 또는 Python::Python 등으로 변경 (pybind11::module이 처리하므로 우선 제거)
)

# OpenMP 링크 (필요한 경우)
if(OPENMP_FOUND)
  target_link_libraries(${PROJECT_NAME}_cpp PRIVATE OpenMP::OpenMP_CXX)
endif()

# 설치 설정
install(TARGETS ${PROJECT_NAME}_cpp
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  RUNTIME DESTINATION bin
)

install(DIRECTORY include/
  DESTINATION include
)

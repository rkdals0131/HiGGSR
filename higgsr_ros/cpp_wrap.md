HiGGSR C++ 성능 가속화 모듈 개발 명세서 (v1.2)
1. 개요
1.1. 목표
본 문서는 HiGGSR ROS2 패키지의 성능 병목 구간을 C++로 구현하여 시스템 전체의 처리 속도를 대폭 향상시키는 것을 목표로 한다. 이 과정은 기존 파이썬 코드의 안정성과 유연성을 해치지 않는 "비침습적 애드온(Non-invasive Add-on)" 방식으로 진행된다.
1.2. 핵심 원칙
탈부착 가능한 구조: C++ 모듈은 선택적으로 빌드 및 사용된다. C++ 모듈이 없거나 빌드에 실패하더라도, 시스템은 자동으로 기존의 순수 파이썬 구현으로 대체 실행(Fallback)되어야 한다.
기존 코드 불변: nodes, visualization 등 상위 레벨의 파이썬 코드는 단 한 줄도 수정하지 않는다.
독립적인 모듈 개발: 성능 병목 지점인 Feature Extraction과 Registration은 서로 의존성 없는 독립적인 C++ 모듈로 개발하여 개별적인 테스트와 적용이 가능하도록 한다.
최고의 성능: 데이터 전달 오버헤드를 최소화하기 위해 pybind11과 Eigen 라이브러리를 적극 활용하여 NumPy 배열을 메모리 복사 없이(Zero-copy) 처리한다.
안정성과 일관성: Python과 C++ 인터페이스 간의 데이터 타입을 명확히 정의하고 검증하여 TypeError, AttributeError와 같은 런타임 에러를 원천적으로 방지한다.
2. 시스템 아키텍처 변경안
C++ 모듈을 비침습적으로 통합하기 위해 다음과 같이 프로젝트 구조를 확장한다.
higgsr_ros/
└── higgsr_ros/
    ├── core/
    │   ├── __init__.py                      # (수정) C++ 모듈 동적 로딩 스위치
    │   ├── feature_extraction.py            # (유지) 순수 파이썬 원본
    │   ├── registration.py                  # (유지) 순수 파이썬 원본
    │   │
    │   ├── cpp_wrappers/                    # (신규) C++ 모듈을 감싸는 파이썬 래퍼 디렉토리
    │   │   ├── __init__.py
    │   │   ├── feature_extraction_wrapper.py
    │   │   └── registration_wrapper.py
    │   │
    │   └── cpp_src/                         # (신규) 모든 C++ 소스 코드
    │       ├── include/                     # (신규) C++ 헤더 파일 디렉토리
    │       │   ├── feature_extraction.hpp
    │       │   └── registration.hpp
    │       ├── feature_extraction.cpp       # 함수 구현부
    │       ├── registration.cpp             # 함수 구현부
    │       └── bindings.cpp                 # Pybind11 인터페이스
    │
    ├── nodes/                             # (변경 없음)
    └── ...


3. 빌드 시스템 수정 계획
현재 higgsr_ros 패키지는 순수 파이썬 패키지(ament_python)입니다. C++ 확장 기능을 안정적으로 통합하기 위해 빌드 시스템을 ament_cmake 기반으로 변경하는 것이 가장 표준적이고 강력한 방법입니다.
3.1. package.xml 수정안
C++ 빌드 의존성과 라이브러리 의존성을 추가합니다.
<?xml version="1.0"?>
<?xml-model href="http://download.ros.org/schema/package_format3.xsd" schematypens="http://www.w3.org/2001/XMLSchema"?>
<package format="3">
  <name>higgsr_ros</name>
  <version>0.2.0</version>
  <description>ROS2 package for HiGGSR localization algorithm with C++ acceleration</description>
  <maintainer email="kikiws70@gmail.com">user1</maintainer>
  <license>Apache License 2.0</license>

  <!-- 빌드 타입 변경을 위한 의존성 -->
  <buildtool_depend>ament_cmake_python</buildtool_depend>
  <buildtool_depend>pybind11_vendor</buildtool_depend>

  <!-- 기존 Python 의존성 -->
  <depend>rclpy</depend>
  <depend>sensor_msgs</depend>
  <depend>geometry_msgs</depend>
  <depend>nav_msgs</depend>
  <depend>std_msgs</depend>
  <depend>visualization_msgs</depend>
  <depend>tf2_ros</depend>
  <depend>tf2_py</depend>
  <depend>higgsr_interface</depend>

  <!-- 신규 C++ 의존성 -->
  <depend>libpcl-dev</depend>
  <depend>libeigen3-dev</depend>

  <test_depend>ament_copyright</test_depend>
  <test_depend>ament_flake8</test_depend>
  <test_depend>ament_pep257</test_depend>
  <test_depend>python3-pytest</test_depend>

  <export>
    <!-- 빌드 타입을 ament_cmake로 변경 -->
    <build_type>ament_cmake</build_type>
  </export>
</package>


3.2. CMakeLists.txt 신규 작성
setup.py는 그대로 유지하되, higgsr_ros 패키지 루트에 CMakeLists.txt 파일을 새로 생성하여 C++ 확장 모듈의 빌드를 담당하게 합니다.
cmake_minimum_required(VERSION 3.8)
project(higgsr_ros)

# 필요한 ROS 2, Python, 외부 라이브러리 탐색
find_package(ament_cmake_auto REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(pybind11_vendor REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(PCL REQUIRED)

# C++ 소스 파일 목록
set(CPP_SOURCES
  higgsr_ros/core/cpp_src/feature_extraction.cpp
  higgsr_ros/core/cpp_src/registration.cpp
  higgsr_ros/core/cpp_src/bindings.cpp
)

# pybind11를 사용한 Python 확장 모듈 빌드
pybind11_add_module(higgsr_core_cpp MODULE ${CPP_SOURCES})

# 타겟에 대한 include 디렉토리 설정
target_include_directories(higgsr_core_cpp PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/higgsr_ros/core/cpp_src/include>
  ${Eigen3_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
)

# 타겟에 대한 라이브러리 링크
target_link_libraries(higgsr_core_cpp PRIVATE
  ${PCL_LIBRARIES}
)

# 컴파일된 모듈을 파이썬 패키지 내부의 올바른 위치로 설치
install(TARGETS higgsr_core_cpp
  DESTINATION higgsr_ros/core
)

# 기존 파이썬 파일들 설치
ament_python_install_package(${PROJECT_NAME})

ament_package()


핵심: 위 CMakeLists.txt는 colcon build 시 higgsr_core_cpp.cpython-3...-x86_64-linux-gnu.so 파일을 생성하고, 이를 파이썬 core 디렉토리 안에 설치하여 파이썬 코드가 from higgsr_ros.core import higgsr_core_cpp 처럼 임포트할 수 있게 해줍니다.
4. 모듈별 C++ 개발 상세 명세
4.1. 모듈 1: feature_extraction_cpp
(이전 버전과 동일: 헤더, 소스 파일 명세)
4.2. 모듈 2: registration_cpp
(이전 버전과 동일: 헤더, 소스 파일 명세)
4.3. Pybind11 인터페이스 상세 명세 (cpp_src/bindings.cpp)
4.3.1. 타입 안정성 및 일관성 확보 방안
Python의 동적 타이핑과 C++의 정적 타이핑 간의 불일치로 인한 런타임 에러를 방지하기 위해 다음 전략을 사용합니다.
py::arg 사용: 모든 인자에 py::arg("arg_name")를 사용하여 파이썬 키워드 인자 호출을 지원하고, 인자 순서에 의한 실수를 방지합니다.
사전 타입 캐스팅 및 검증: 복잡한 파이썬 객체(py::list, py::dict)는 C++ 내부에서 사용하기 전에 타입을 명시적으로 캐스팅하고 유효성을 검사하는 헬퍼 함수를 만듭니다.
예시: level_configs 파싱
// registration.cpp 내부에 추가될 수 있는 헬퍼 함수
#include <stdexcept>

struct LevelConfig {
    std::vector<int> grid_division;
    std::string search_area_type;
    // ... 기타 설정값
};

std::vector<LevelConfig> parseLevelConfigs(const py::list& level_configs_py) {
    std::vector<LevelConfig> configs;
    for (const auto& item : level_configs_py) {
        py::dict config_dict = item.cast<py::dict>();
        LevelConfig cfg;

        if (!config_dict.contains("grid_division") || !config_dict.contains("search_area_type")) {
            throw std::runtime_error("Invalid level_config: missing required keys.");
        }
        cfg.grid_division = config_dict["grid_division"].cast<std::vector<int>>();
        cfg.search_area_type = config_dict["search_area_type"].cast<std::string>();
        // ... 나머지 필드도 동일하게 캐스팅 및 검증
        configs.push_back(cfg);
    }
    return configs;
}

// findBestTransform 함수 내부에서 호출
// TransformResult findBestTransform(...) {
//     try {
//         auto cpp_level_configs = parseLevelConfigs(level_configs_py);
//         // ... 이후 로직
//     } catch (const std::runtime_error& e) {
//         // 에러 발생 시 Python 예외로 변환하여 전달
//         throw py::value_error(e.what());
//     }
// }


4.3.2. 최종 바인딩 코드 (bindings.cpp)
(이전 버전과 동일하되, findBestTransform 내부에서 위와 같은 예외 처리 로직이 포함되어야 함)
5. 단계적 개발 및 검증 계획
각 기능을 한 번에 개발하지 않고, 작은 단위로 구현하고 즉시 검증하여 안정성을 확보합니다.
1단계: 빌드 시스템 및 빈 모듈 통합
구현: package.xml과 CMakeLists.txt를 위 명세대로 수정/생성합니다. cpp_src 디렉토리와 빈 .cpp, .hpp 파일들을 만듭니다. bindings.cpp에는 PYBIND11_MODULE 매크로와 비어있는 내용만 둡니다.
검증:
colcon build --packages-select higgsr_ros를 실행하여 에러 없이 빌드되는지 확인합니다.
install/higgsr_ros/lib/python3.10/site-packages/higgsr_ros/core/ 디렉토리 안에 higgsr_core_cpp...so 파일이 생성되었는지 확인합니다.
파이썬 터미널에서 from higgsr_ros.core import higgsr_core_cpp를 실행하여 임포트 에러가 없는지 확인합니다.
이 단계가 성공하면, Python-C++ 연동을 위한 가장 큰 허들을 넘은 것입니다.
2단계: feature_extraction_cpp 구현 및 검증
구현: 명세에 따라 feature_extraction.hpp, .cpp, 그리고 bindings.cpp에 해당 함수 바인딩 부분을 구현합니다.
검증:
별도의 파이썬 테스트 스크립트(test_feature_extraction.py)를 작성합니다.
이 스크립트에서 동일한 입력(밀도 맵 NumPy 배열 등)을 (A) 기존 파이썬 함수와 (B) 새로 만든 C++ 모듈 함수에 각각 전달합니다.
np.testing.assert_allclose()를 사용하여 두 결과(키포인트 배열)가 거의 일치하는지 검증합니다.
timeit 모듈로 두 함수의 실행 시간을 측정하여 성능 향상을 정량적으로 확인합니다.
3단계: registration_cpp 구현 및 검증
구현: 명세에 따라 registration.hpp, .cpp, 그리고 bindings.cpp에 해당 함수 바인딩 부분을 구현합니다. OpenMP 병렬화는 일단 비활성화하고 단일 스레드로 구현하여 로직 검증에 집중합니다.
검증:
test_registration.py 스크립트를 작성합니다.
2단계와 동일하게, 같은 키포인트 배열을 파이썬 함수와 C++ 함수에 각각 전달합니다.
결과로 나온 tx, ty, theta_deg, score 값이 허용 오차 내에서 일치하는지 검증합니다.
성능을 측정합니다. (단일 스레드에서도 Python보다 훨씬 빠를 것입니다.)
로직 검증이 완료되면, OpenMP 병렬화를 활성화하고 성능이 추가로 향상되는지 확인합니다.
4.단계: 최종 통합 및 시스템 테스트
구현: cpp_wrappers 디렉토리와 core/__init__.py 수정안을 최종적으로 적용합니다.
검증:
C++ 모듈이 있는 상태에서 ros2 run higgsr_ros file_processor_node ... 또는 test_rviz_visualization.py를 실행하여 전체 시스템이 정상 동작하는지 확인합니다. 터미널에 "Using C++ accelerated..." 메시지가 출력되어야 합니다.
install 디렉토리에서 higgsr_core_cpp...so 파일의 이름을 잠시 바꾸거나 삭제하여 C++ 모듈이 없는 상태를 강제로 만듭니다.
다시 시스템을 실행하여 "Falling back to Python..." 메시지가 출력되고, 속도는 느려지지만 기능적으로는 동일하게 동작하는지 확인합니다.
이 Fallback 테스트가 성공하면, 비침습적 애드온 구조가 완벽하게 구현된 것입니다.

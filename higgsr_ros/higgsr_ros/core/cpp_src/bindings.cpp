#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>
#include "include/feature_extraction.hpp"
#include "include/registration.hpp"

namespace py = pybind11;

// 타입 안전성을 위한 헬퍼 함수들
namespace {

/**
 * @brief Python 리스트에서 LevelConfig 벡터로 안전하게 변환
 */
std::vector<higgsr_core::LevelConfig> parseLevelConfigs(const py::list& level_configs_py) {
    std::vector<higgsr_core::LevelConfig> configs;
    configs.reserve(level_configs_py.size());
    
    for (size_t i = 0; i < level_configs_py.size(); ++i) {
        try {
            py::dict config_dict = level_configs_py[i].cast<py::dict>();
            higgsr_core::LevelConfig cfg;
            
            // 필수 키 존재 확인
            if (!config_dict.contains("grid_division")) {
                throw std::runtime_error("Missing required key 'grid_division' in level config " + std::to_string(i));
            }
            if (!config_dict.contains("search_area_type")) {
                throw std::runtime_error("Missing required key 'search_area_type' in level config " + std::to_string(i));
            }
            if (!config_dict.contains("tx_ty_search_steps_per_cell")) {
                throw std::runtime_error("Missing required key 'tx_ty_search_steps_per_cell' in level config " + std::to_string(i));
            }
            
            // 안전한 타입 캐스팅
            cfg.grid_division = config_dict["grid_division"].cast<std::vector<int>>();
            cfg.search_area_type = config_dict["search_area_type"].cast<std::string>();
            cfg.tx_ty_search_steps = config_dict["tx_ty_search_steps_per_cell"].cast<std::vector<int>>();
            
            // 선택적 키들
            if (config_dict.contains("correspondence_dist_thresh_factor")) {
                cfg.correspondence_dist_thresh_factor = 
                    config_dict["correspondence_dist_thresh_factor"].cast<double>();
            }
            
            // 유효성 검증
            if (!cfg.isValid()) {
                throw std::runtime_error("Invalid level config at index " + std::to_string(i));
            }
            
            configs.push_back(cfg);
            
        } catch (const py::cast_error& e) {
            throw std::runtime_error("Type casting error in level config " + std::to_string(i) + ": " + e.what());
        }
    }
    
    return configs;
}

/**
 * @brief NumPy 배열에서 키포인트 벡터로 안전하게 변환
 */
std::vector<higgsr_core::Keypoint> numpyToKeypoints(const py::array_t<double>& keypoints_array) {
    // 배열 형태 검증
    if (keypoints_array.ndim() != 2) {
        throw std::invalid_argument("keypoints array must be 2-dimensional");
    }
    if (keypoints_array.shape(1) != 2) {
        throw std::invalid_argument("keypoints array must have 2 columns (x, y)");
    }
    
    std::vector<higgsr_core::Keypoint> keypoints;
    keypoints.reserve(keypoints_array.shape(0));
    
    auto buf = keypoints_array.request();
    double* ptr = static_cast<double*>(buf.ptr);
    
    for (py::ssize_t i = 0; i < keypoints_array.shape(0); ++i) {
        double x = ptr[i * 2 + 0];
        double y = ptr[i * 2 + 1];
        
        // 유효성 검증
        if (!std::isfinite(x) || !std::isfinite(y)) {
            continue;  // 유효하지 않은 키포인트는 건너뛰기
        }
        
        keypoints.emplace_back(x, y);
    }
    
    return keypoints;
}

/**
 * @brief 키포인트 벡터에서 NumPy 배열로 안전하게 변환
 */
py::array_t<double> keypointsToNumpy(const std::vector<higgsr_core::Keypoint>& keypoints) {
    if (keypoints.empty()) {
        return py::array_t<double>(py::array::ShapeContainer({0, 2}));
    }
    
    auto result = py::array_t<double>(
        py::array::ShapeContainer({static_cast<py::ssize_t>(keypoints.size()), 2})
    );
    
    auto buf = result.request();
    double* ptr = static_cast<double*>(buf.ptr);
    
    for (size_t i = 0; i < keypoints.size(); ++i) {
        ptr[i * 2 + 0] = keypoints[i].x;
        ptr[i * 2 + 1] = keypoints[i].y;
    }
    
    return result;
}

} // anonymous namespace

// Python 바인딩 래퍼 함수들
namespace higgsr_core {

/**
 * @brief Feature extraction을 위한 Python 래퍼 함수
 */
py::array_t<double> extractHighDensityKeypointsPython(
    const py::array_t<double>& density_map,
    const py::array_t<double>& x_edges,
    const py::array_t<double>& y_edges,
    double density_threshold
) {
    try {
        // NumPy 배열 유효성 검증
        if (density_map.ndim() != 2) {
            throw py::value_error("density_map must be 2-dimensional");
        }
        if (x_edges.ndim() != 1 || y_edges.ndim() != 1) {
            throw py::value_error("x_edges and y_edges must be 1-dimensional");
        }
        
        // 버퍼 요청
        auto density_buf = density_map.request();
        auto x_buf = x_edges.request();
        auto y_buf = y_edges.request();
        
        // 파라미터 설정
        FeatureExtractionParams params;
        params.density_threshold = density_threshold;
        
        // C++ 함수 호출
        auto keypoints = extractHighDensityKeypoints(
            static_cast<double*>(density_buf.ptr),
            static_cast<int>(density_map.shape(0)),
            static_cast<int>(density_map.shape(1)),
            static_cast<double*>(x_buf.ptr),
            static_cast<double*>(y_buf.ptr),
            static_cast<int>(x_edges.size()),
            static_cast<int>(y_edges.size()),
            params
        );
        
        // NumPy 배열로 변환
        return keypointsToNumpy(keypoints);
        
    } catch (const std::exception& e) {
        throw py::value_error("Feature extraction error: " + std::string(e.what()));
    }
}

/**
 * @brief 키포인트 변환을 위한 Python 래퍼 함수
 */
py::array_t<double> applyTransformToKeypointsPython(
    const py::array_t<double>& keypoints_array,
    double tx,
    double ty,
    double theta_rad
) {
    try {
        // 키포인트 변환
        auto keypoints = numpyToKeypoints(keypoints_array);
        
        // C++ 함수 호출
        auto transformed_keypoints = applyTransformToKeypoints(keypoints, tx, ty, theta_rad);
        
        // NumPy 배열로 변환
        return keypointsToNumpy(transformed_keypoints);
        
    } catch (const std::exception& e) {
        throw py::value_error("Keypoint transformation error: " + std::string(e.what()));
    }
}

/**
 * @brief 계층적 적응 탐색을 위한 Python 래퍼 함수 - 5개 원소 튜플 반환
 */
py::tuple hierarchicalAdaptiveSearchPython(
    const py::array_t<double>& global_map_keypoints,
    const py::array_t<double>& live_scan_keypoints,
    const py::array_t<double>& initial_map_x_edges,
    const py::array_t<double>& initial_map_y_edges,
    const py::list& level_configs,
    int num_candidates_to_select_per_level = 5,
    double min_candidate_separation_factor = 2.0,
    double base_grid_cell_size = 1.0,
    int num_processes = 0
) {
    try {
        // 입력 데이터 변환
        auto global_keypoints = numpyToKeypoints(global_map_keypoints);
        auto scan_keypoints = numpyToKeypoints(live_scan_keypoints);
        
        // 경계값 변환
        std::vector<double> x_edges_vec, y_edges_vec;
        
        auto x_buf = initial_map_x_edges.request();
        auto y_buf = initial_map_y_edges.request();
        
        double* x_ptr = static_cast<double*>(x_buf.ptr);
        double* y_ptr = static_cast<double*>(y_buf.ptr);
        
        x_edges_vec.assign(x_ptr, x_ptr + initial_map_x_edges.size());
        y_edges_vec.assign(y_ptr, y_ptr + initial_map_y_edges.size());
        
        // 파라미터 설정
        HierarchicalSearchParams params;
        params.level_configs = parseLevelConfigs(level_configs);
        params.num_candidates_to_select_per_level = num_candidates_to_select_per_level;
        params.min_candidate_separation_factor = min_candidate_separation_factor;
        params.base_grid_cell_size = base_grid_cell_size;
        params.num_processes = num_processes;
        
        // 파라미터 유효성 검증
        if (!params.isValid()) {
            throw py::value_error("Invalid hierarchical search parameters");
        }
        
        // C++ 함수 호출
        auto result = hierarchicalAdaptiveSearch(
            global_keypoints, scan_keypoints, 
            x_edges_vec, y_edges_vec, params
        );
        
        // Python dict 생성 (첫 번째 반환값)
        py::dict result_dict;
        result_dict["tx"] = result.tx;
        result_dict["ty"] = result.ty;
        result_dict["theta_deg"] = result.theta_deg;
        result_dict["score"] = result.score;
        
        // 시각화 데이터는 C++에서 생성하지 않으므로 빈 리스트로 반환
        py::list viz_data;
        
        // 🐍 5개 원소를 가진 튜플을 생성하여 반환
        return py::make_tuple(
            result_dict,                                    // 최종 변환 결과 dict
            result.score,                                   // 최종 점수
            viz_data,                                       // 시각화 데이터 (빈 리스트)
            result.execution_time_ms / 1000.0,              // 실행 시간 (초)
            result.iterations                               // 총 반복 횟수
        );
        
    } catch (const std::exception& e) {
        throw py::value_error("Hierarchical search error: " + std::string(e.what()));
    }
}

} // namespace higgsr_core

// 모듈 정의
PYBIND11_MODULE(higgsr_core_cpp, m) {
    m.doc() = "HiGGSR C++ Performance Acceleration Module";
    
    // 구조체 바인딩
    py::class_<higgsr_core::Keypoint>(m, "Keypoint")
        .def(py::init<>())
        .def(py::init<double, double>())
        .def_readwrite("x", &higgsr_core::Keypoint::x)
        .def_readwrite("y", &higgsr_core::Keypoint::y);
    
    py::class_<higgsr_core::TransformResult>(m, "TransformResult")
        .def(py::init<>())
        .def(py::init<double, double, double, double, int, double>())
        .def_readwrite("tx", &higgsr_core::TransformResult::tx)
        .def_readwrite("ty", &higgsr_core::TransformResult::ty)
        .def_readwrite("theta_deg", &higgsr_core::TransformResult::theta_deg)
        .def_readwrite("score", &higgsr_core::TransformResult::score)
        .def_readwrite("iterations", &higgsr_core::TransformResult::iterations)
        .def_readwrite("success", &higgsr_core::TransformResult::success)
        .def_readwrite("execution_time_ms", &higgsr_core::TransformResult::execution_time_ms)
        .def("isValid", &higgsr_core::TransformResult::isValid);
    
    // Feature extraction 함수들
    m.def("extract_high_density_keypoints", 
          &higgsr_core::extractHighDensityKeypointsPython,
          "Extract high density keypoints from density map",
          py::arg("density_map"), 
          py::arg("x_edges"), 
          py::arg("y_edges"), 
          py::arg("density_threshold"));
    
    m.def("apply_transform_to_keypoints", 
          &higgsr_core::applyTransformToKeypointsPython,
          "Apply 2D transformation to keypoints",
          py::arg("keypoints"), 
          py::arg("tx"), 
          py::arg("ty"), 
          py::arg("theta_rad"));
    
    // Registration 함수들
    m.def("hierarchical_adaptive_search", 
          &higgsr_core::hierarchicalAdaptiveSearchPython,
          "Perform hierarchical adaptive search for registration",
          py::arg("global_map_keypoints"),
          py::arg("live_scan_keypoints"),
          py::arg("initial_map_x_edges"),
          py::arg("initial_map_y_edges"),
          py::arg("level_configs"),
          py::arg("num_candidates_to_select_per_level") = 5,
          py::arg("min_candidate_separation_factor") = 2.0,
          py::arg("base_grid_cell_size") = 1.0,
          py::arg("num_processes") = 0);
    
    // 유틸리티 함수들
    m.def("count_correspondences_kdtree", 
          [](const py::array_t<double>& transformed_keypoints,
             const py::array_t<double>& global_map_keypoints,
             double distance_threshold) {
              auto transformed_kps = numpyToKeypoints(transformed_keypoints);
              auto global_kps = numpyToKeypoints(global_map_keypoints);
              return higgsr_core::countCorrespondencesKDTree(transformed_kps, global_kps, distance_threshold);
          },
          "Count correspondences using KDTree",
          py::arg("transformed_keypoints"),
          py::arg("global_map_keypoints"),
          py::arg("distance_threshold"));
    
    // 버전 정보
    m.attr("__version__") = "1.0.0";
    m.attr("__author__") = "HiGGSR Development Team";
} 
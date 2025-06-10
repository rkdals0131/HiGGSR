#ifndef HIGGSR_REGISTRATION_HPP
#define HIGGSR_REGISTRATION_HPP

#include <vector>
#include <string>
#include <memory>
#include <Eigen/Dense>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/kdtree/kdtree_flann.h>
#include "feature_extraction.hpp"

namespace higgsr_core {

/**
 * @brief 변환 결과를 나타내는 구조체
 */
struct TransformResult {
    double tx;           // X 방향 이동
    double ty;           // Y 방향 이동  
    double theta_deg;    // 회전 각도(도)
    double score;        // 매칭 점수
    int iterations;      // 수행된 반복 횟수
    bool success;        // 성공 여부
    double execution_time_ms; // 실행 시간 (밀리초)
    
    TransformResult() 
        : tx(0.0), ty(0.0), theta_deg(0.0), score(-1.0), 
          iterations(0), success(false), execution_time_ms(0.0) {}
          
    TransformResult(double tx_val, double ty_val, double theta_deg_val, 
                   double score_val, int iter_val = 0, double exec_time_ms = 0.0)
        : tx(tx_val), ty(ty_val), theta_deg(theta_deg_val), 
          score(score_val), iterations(iter_val), success(score_val >= 0.0), 
          execution_time_ms(exec_time_ms) {}
          
    // 결과 유효성 검증
    bool isValid() const {
        return success && std::isfinite(tx) && std::isfinite(ty) && 
               std::isfinite(theta_deg) && std::isfinite(score);
    }
};

/**
 * @brief 레벨 구성 파라미터
 */
struct LevelConfig {
    std::vector<int> grid_division;           // [num_x, num_y]
    std::string search_area_type;             // "full_map", "adaptive", etc.
    std::vector<int> tx_ty_search_steps;      // [tx_steps, ty_steps]
    double correspondence_dist_thresh_factor; // 매칭 거리 임계값 팩터
    
    LevelConfig() 
        : grid_division({1, 1}), search_area_type("full_map"),
          tx_ty_search_steps({10, 10}), correspondence_dist_thresh_factor(1.5) {}
          
    // 파라미터 유효성 검증
    bool isValid() const {
        return grid_division.size() == 2 && grid_division[0] > 0 && grid_division[1] > 0 &&
               tx_ty_search_steps.size() == 2 && tx_ty_search_steps[0] > 0 && tx_ty_search_steps[1] > 0 &&
               correspondence_dist_thresh_factor > 0.0 &&
               !search_area_type.empty();
    }
};

/**
 * @brief 계층적 적응 탐색 파라미터
 */
struct HierarchicalSearchParams {
    std::vector<LevelConfig> level_configs;
    int num_candidates_to_select_per_level;
    double min_candidate_separation_factor;
    double base_grid_cell_size;
    int num_processes;
    double global_correspondence_threshold;
    
    HierarchicalSearchParams()
        : num_candidates_to_select_per_level(5),
          min_candidate_separation_factor(2.0),
          base_grid_cell_size(1.0),
          num_processes(0),
          global_correspondence_threshold(0.5) {}
          
    // 파라미터 유효성 검증
    bool isValid() const {
        if (level_configs.empty()) return false;
        for (const auto& config : level_configs) {
            if (!config.isValid()) return false;
        }
        return num_candidates_to_select_per_level > 0 &&
               min_candidate_separation_factor > 0.0 &&
               base_grid_cell_size > 0.0 &&
               num_processes >= 0 &&
               global_correspondence_threshold > 0.0;
    }
};

/**
 * @brief 후보 정보 구조체
 */
struct CandidateInfo {
    double score;
    double tx;
    double ty; 
    double theta_deg;
    double center_x;
    double center_y;
    
    CandidateInfo()
        : score(-1.0), tx(0.0), ty(0.0), theta_deg(0.0),
          center_x(0.0), center_y(0.0) {}
          
    CandidateInfo(double s, double t_x, double t_y, double theta, double cx, double cy)
        : score(s), tx(t_x), ty(t_y), theta_deg(theta), center_x(cx), center_y(cy) {}
        
    bool isValid() const {
        return score >= 0.0 && std::isfinite(tx) && std::isfinite(ty) && 
               std::isfinite(theta_deg) && std::isfinite(center_x) && std::isfinite(center_y);
    }
};

/**
 * @brief KDTree를 사용한 대응점 계산
 * 
 * @param transformed_keypoints 변환된 키포인트들
 * @param global_map_keypoints 글로벌 맵 키포인트들
 * @param distance_threshold 매칭 거리 임계값
 * @return 매칭된 대응점 개수
 */
int countCorrespondencesKDTree(
    const std::vector<Keypoint>& transformed_keypoints,
    const std::vector<Keypoint>& global_map_keypoints, 
    double distance_threshold
);

/**
 * @brief 다양한 후보 선택 함수
 * 
 * @param candidates_info 모든 후보 정보
 * @param num_to_select 선택할 후보 개수
 * @param separation_factor 최소 분리 거리 팩터
 * @param cell_size_x X 방향 셀 크기
 * @param cell_size_y Y 방향 셀 크기
 * @param map_x_range X 방향 맵 범위 [min, max]
 * @param map_y_range Y 방향 맵 범위 [min, max]
 * @return 선택된 후보들
 */
std::vector<CandidateInfo> selectDiverseCandidates(
    const std::vector<CandidateInfo>& candidates_info,
    int num_to_select,
    double separation_factor,
    double cell_size_x,
    double cell_size_y,
    const std::pair<double, double>& map_x_range,
    const std::pair<double, double>& map_y_range
);

/**
 * @brief 계층적 적응 탐색 메인 함수
 * 
 * @param global_map_keypoints 글로벌 맵의 키포인트들
 * @param live_scan_keypoints 현재 스캔의 키포인트들  
 * @param initial_map_x_edges 초기 맵 X 경계
 * @param initial_map_y_edges 초기 맵 Y 경계
 * @param params 탐색 파라미터
 * @return 최적 변환 결과
 * 
 * @throws std::invalid_argument 입력이 유효하지 않은 경우
 * @throws std::runtime_error 탐색 중 오류가 발생한 경우
 */
TransformResult hierarchicalAdaptiveSearch(
    const std::vector<Keypoint>& global_map_keypoints,
    const std::vector<Keypoint>& live_scan_keypoints,
    const std::vector<double>& initial_map_x_edges,
    const std::vector<double>& initial_map_y_edges,
    const HierarchicalSearchParams& params
);

// TODO: OpenMP 병렬화 함수들 추가 예정
// - parallelSearchInSuperGrids()
// - optimizeTransformationICP()
// - refineTransformationResults()

// TODO: PCL 기반 고급 기능들 추가 예정  
// - pcl::KdTreeFLANN 최적화
// - pcl::CorrespondenceEstimation 활용
// - pcl::TransformationEstimation 통합

/**
 * @brief 입력 데이터 유효성 검증 헬퍼 함수
 */
bool validateInputData(
    const std::vector<Keypoint>& global_keypoints,
    const std::vector<Keypoint>& scan_keypoints,
    const std::vector<double>& x_edges,
    const std::vector<double>& y_edges,
    const HierarchicalSearchParams& params
);

} // namespace higgsr_core

#endif // HIGGSR_REGISTRATION_HPP 
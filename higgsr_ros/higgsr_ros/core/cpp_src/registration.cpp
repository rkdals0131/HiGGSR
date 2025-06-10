#include "include/registration.hpp"
#include <stdexcept>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <limits>
#include <unordered_set>

namespace higgsr_core {

bool validateInputData(
    const std::vector<Keypoint>& global_keypoints,
    const std::vector<Keypoint>& scan_keypoints,
    const std::vector<double>& x_edges,
    const std::vector<double>& y_edges,
    const HierarchicalSearchParams& params
) {
    // 키포인트 유효성 검증
    if (global_keypoints.empty()) {
        std::cerr << "ERROR: global_keypoints is empty" << std::endl;
        return false;
    }
    if (scan_keypoints.empty()) {
        std::cerr << "ERROR: scan_keypoints is empty" << std::endl;
        return false;
    }
    
    // 경계값 유효성 검증
    if (x_edges.size() < 2 || y_edges.size() < 2) {
        std::cerr << "ERROR: edge arrays must have at least 2 elements" << std::endl;
        return false;
    }
    
    // 경계값 정렬 확인
    for (size_t i = 1; i < x_edges.size(); ++i) {
        if (x_edges[i] <= x_edges[i-1]) {
            std::cerr << "ERROR: x_edges must be in ascending order" << std::endl;
            return false;
        }
    }
    for (size_t i = 1; i < y_edges.size(); ++i) {
        if (y_edges[i] <= y_edges[i-1]) {
            std::cerr << "ERROR: y_edges must be in ascending order" << std::endl;
            return false;
        }
    }
    
    // 파라미터 유효성 검증
    if (!params.isValid()) {
        std::cerr << "ERROR: invalid hierarchical search parameters" << std::endl;
        return false;
    }
    
    // 키포인트 값 유효성 검증
    for (const auto& kp : global_keypoints) {
        if (!std::isfinite(kp.x) || !std::isfinite(kp.y)) {
            std::cerr << "ERROR: global_keypoints contains non-finite values" << std::endl;
            return false;
        }
    }
    for (const auto& kp : scan_keypoints) {
        if (!std::isfinite(kp.x) || !std::isfinite(kp.y)) {
            std::cerr << "ERROR: scan_keypoints contains non-finite values" << std::endl;
            return false;
        }
    }
    
    return true;
}

int countCorrespondencesKDTree(
    const std::vector<Keypoint>& transformed_keypoints,
    const std::vector<Keypoint>& global_map_keypoints, 
    double distance_threshold
) {
    // TODO: 실제 KDTree 기반 구현 예정
    // 현재는 단순한 브루트포스 방식의 플레이스홀더
    
    // 입력 유효성 검증
    if (transformed_keypoints.empty() || global_map_keypoints.empty()) {
        return 0;
    }
    if (distance_threshold <= 0.0 || !std::isfinite(distance_threshold)) {
        throw std::invalid_argument("distance_threshold must be positive and finite");
    }
    
    int correspondence_count = 0;
    double threshold_squared = distance_threshold * distance_threshold;
    
    try {
        // TODO: PCL KdTreeFLANN 또는 Eigen 기반 KDTree 사용 예정
        // 임시 플레이스홀더: O(N*M) 브루트포스 방식
        for (const auto& transformed_kp : transformed_keypoints) {
            bool found_correspondence = false;
            
            for (const auto& global_kp : global_map_keypoints) {
                double dx = transformed_kp.x - global_kp.x;
                double dy = transformed_kp.y - global_kp.y;
                double distance_squared = dx * dx + dy * dy;
                
                if (distance_squared <= threshold_squared) {
                    found_correspondence = true;
                    break;  // 첫 번째 매칭만 카운트
                }
            }
            
            if (found_correspondence) {
                correspondence_count++;
            }
        }
        
    } catch (const std::exception& e) {
        throw std::runtime_error("Error during correspondence counting: " + std::string(e.what()));
    }
    
    return correspondence_count;
}

std::vector<CandidateInfo> selectDiverseCandidates(
    const std::vector<CandidateInfo>& candidates_info,
    int num_to_select,
    double separation_factor,
    double cell_size_x,
    double cell_size_y,
    const std::pair<double, double>& map_x_range,
    const std::pair<double, double>& map_y_range
) {
    // TODO: 실제 구현 예정
    // 현재는 타입 안전성과 기본 로직만 구현
    
    // 입력 유효성 검증
    if (num_to_select <= 0) {
        throw std::invalid_argument("num_to_select must be positive");
    }
    if (separation_factor <= 0.0 || !std::isfinite(separation_factor)) {
        throw std::invalid_argument("separation_factor must be positive and finite");
    }
    if (cell_size_x <= 0.0 || cell_size_y <= 0.0) {
        throw std::invalid_argument("cell sizes must be positive");
    }
    
    // 유효한 후보들만 필터링
    std::vector<CandidateInfo> valid_candidates;
    for (const auto& candidate : candidates_info) {
        if (candidate.isValid() && candidate.score > -std::numeric_limits<double>::infinity()) {
            valid_candidates.push_back(candidate);
        }
    }
    
    if (valid_candidates.empty()) {
        std::cout << "WARNING: No valid candidates found" << std::endl;
        return {};
    }
    
    // 점수 기준으로 정렬
    std::sort(valid_candidates.begin(), valid_candidates.end(),
        [](const CandidateInfo& a, const CandidateInfo& b) {
            return a.score > b.score;  // 내림차순
        });
    
    std::vector<CandidateInfo> selected_candidates;
    selected_candidates.reserve(num_to_select);
    
    double min_separation_x = separation_factor * cell_size_x;
    double min_separation_y = separation_factor * cell_size_y;
    
    try {
        for (const auto& candidate : valid_candidates) {
            if (static_cast<int>(selected_candidates.size()) >= num_to_select) {
                break;
            }
            
            // 기존 선택된 후보들과의 거리 체크
            bool is_far_enough = true;
            for (const auto& selected : selected_candidates) {
                double dx = std::abs(candidate.center_x - selected.center_x);
                double dy = std::abs(candidate.center_y - selected.center_y);
                
                if (dx < min_separation_x && dy < min_separation_y) {
                    is_far_enough = false;
                    break;
                }
            }
            
            if (is_far_enough) {
                selected_candidates.push_back(candidate);
            }
        }
        
        std::cout << "INFO: Selected " << selected_candidates.size() 
                  << " diverse candidates from " << valid_candidates.size() 
                  << " valid candidates (C++ implementation)" << std::endl;
                  
    } catch (const std::exception& e) {
        throw std::runtime_error("Error during candidate selection: " + std::string(e.what()));
    }
    
    return selected_candidates;
}

TransformResult hierarchicalAdaptiveSearch(
    const std::vector<Keypoint>& global_map_keypoints,
    const std::vector<Keypoint>& live_scan_keypoints,
    const std::vector<double>& initial_map_x_edges,
    const std::vector<double>& initial_map_y_edges,
    const HierarchicalSearchParams& params
) {
    // TODO: 실제 계층적 적응 탐색 구현 예정
    // 현재는 타입 안전성과 기본 구조만 구현
    
    std::cout << "INFO: Starting hierarchical adaptive search (C++ implementation)" << std::endl;
    
    // 입력 데이터 유효성 검증
    if (!validateInputData(global_map_keypoints, live_scan_keypoints, 
                          initial_map_x_edges, initial_map_y_edges, params)) {
        throw std::invalid_argument("Invalid input data for hierarchical search");
    }
    
    TransformResult best_result;
    
    try {
        // TODO: 실제 계층적 탐색 로직 구현
        // 임시 플레이스홀더: 기본값 반환
        
        std::cout << "INFO: Processing " << params.level_configs.size() 
                  << " levels with " << global_map_keypoints.size() 
                  << " global keypoints and " << live_scan_keypoints.size() 
                  << " scan keypoints" << std::endl;
        
        // 각 레벨별 탐색 시뮬레이션
        for (size_t level_idx = 0; level_idx < params.level_configs.size(); ++level_idx) {
            const auto& level_config = params.level_configs[level_idx];
            
            std::cout << "  Level " << (level_idx + 1) << "/" << params.level_configs.size()
                      << ": Grid division [" << level_config.grid_division[0] 
                      << ", " << level_config.grid_division[1] << "]" << std::endl;
            
            // TODO: 실제 레벨별 탐색 로직
            // - 그리드 분할
            // - 각 셀에서 변환 파라미터 탐색
            // - KDTree 기반 대응점 계산
            // - 최적 후보 선택
        }
        
        // 임시 결과 (실제 구현 시 제거)
        best_result = TransformResult(0.0, 0.0, 0.0, 0.0, 100);
        best_result.success = false;  // 플레이스홀더이므로 실패로 마킹
        
        std::cout << "INFO: Hierarchical search completed (placeholder implementation)" << std::endl;
        
    } catch (const std::exception& e) {
        throw std::runtime_error("Error during hierarchical search: " + std::string(e.what()));
    }
    
    return best_result;
}

// TODO: 향후 추가될 고급 함수들의 플레이스홀더
/*
TransformResult parallelSearchInSuperGrids(
    const std::vector<Keypoint>& global_keypoints,
    const std::vector<Keypoint>& scan_keypoints,
    const LevelConfig& config,
    int num_threads
) {
    // TODO: OpenMP 기반 병렬 탐색 구현
    TransformResult result;
    return result;
}

TransformResult optimizeTransformationICP(
    const std::vector<Keypoint>& global_keypoints,
    const std::vector<Keypoint>& scan_keypoints,
    const TransformResult& initial_guess,
    int max_iterations
) {
    // TODO: ICP 기반 변환 최적화
    TransformResult result;
    return result;
}

std::vector<TransformResult> refineTransformationResults(
    const std::vector<TransformResult>& initial_results,
    const std::vector<Keypoint>& global_keypoints,
    const std::vector<Keypoint>& scan_keypoints
) {
    // TODO: 결과 세밀화 및 검증
    std::vector<TransformResult> refined_results;
    return refined_results;
}
*/

} // namespace higgsr_core 
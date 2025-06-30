#include "include/registration.hpp"
#include <stdexcept>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <limits>
#include <unordered_set>
#include <chrono>
#include <vector>
#include <iomanip>

// 🚀 OpenMP for 멀티스레드 병렬화
#ifdef _OPENMP
#include <omp.h>
#endif

// M_PI 정의 (일부 컴파일러에서 누락될 수 있음)
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

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

/**
 * @brief count_correspondences_kdtree 함수의 C++ 구현 - 파이썬 버전과 동일한 로직
 */
int countCorrespondencesKDTree(
    const std::vector<Keypoint>& transformed_keypoints,
    const std::vector<Keypoint>& global_map_keypoints, 
    double distance_threshold
) {
    // 입력 유효성 검증 - 파이썬과 동일
    if (transformed_keypoints.empty() || global_map_keypoints.empty()) {
        return 0;
    }
    if (distance_threshold <= 0.0 || !std::isfinite(distance_threshold)) {
        throw std::invalid_argument("distance_threshold must be positive and finite");
    }
    
    const double threshold_squared = distance_threshold * distance_threshold;
    const size_t num_transformed = transformed_keypoints.size();
    const size_t num_global = global_map_keypoints.size();
    
    // 🚀 병렬 카운팅 (OpenMP 사용) - 파이썬의 scipy KDTree와 동일한 결과
    int correspondence_count = 0;
    
    #ifdef _OPENMP
    #pragma omp parallel for reduction(+:correspondence_count) schedule(dynamic)
    for (size_t i = 0; i < num_transformed; ++i) {
        const auto& transformed_kp = transformed_keypoints[i];
        
        // 각 변환된 키포인트에 대해 가장 가까운 글로벌 키포인트 찾기
        double min_dist_squared = std::numeric_limits<double>::max();
        
        for (size_t j = 0; j < num_global; ++j) {
            const auto& global_kp = global_map_keypoints[j];
            const double dx = transformed_kp.x - global_kp.x;
            const double dy = transformed_kp.y - global_kp.y;
            const double dist_squared = dx * dx + dy * dy;
            
            if (dist_squared < min_dist_squared) {
                min_dist_squared = dist_squared;
            }
        }
        
        // 임계값 이내인 매칭이 있으면 카운트 증가 (파이썬과 동일)
        if (min_dist_squared <= threshold_squared) {
            correspondence_count++;
        }
    }
    #else
    // 싱글스레드 버전
    for (const auto& transformed_kp : transformed_keypoints) {
        double min_dist_squared = std::numeric_limits<double>::max();
        
        for (const auto& global_kp : global_map_keypoints) {
            const double dx = transformed_kp.x - global_kp.x;
            const double dy = transformed_kp.y - global_kp.y;
            const double distance_squared = dx * dx + dy * dy;
            
            if (distance_squared < min_dist_squared) {
                min_dist_squared = distance_squared;
            }
        }
        
        if (min_dist_squared <= threshold_squared) {
            correspondence_count++;
        }
    }
    #endif
    
    return correspondence_count;
}

/**
 * @brief process_super_grid_cell 함수의 C++ 구현 - 파이썬과 동일한 로직
 */
CandidateInfo processSuperGridCell(
    double super_cx, double super_cy,
    const std::vector<Keypoint>& live_scan_keypoints,
    const std::vector<Keypoint>& global_map_keypoints,
    double actual_local_search_tx_half_width,
    double actual_local_search_ty_half_width,
    int tx_search_steps, int ty_search_steps,
    const std::vector<double>& theta_candidates_rad,
    const std::vector<double>& theta_candidates_deg,
    double actual_correspondence_dist_thresh
) {
    // tx_candidates 생성 - 파이썬의 np.linspace와 동일
    std::vector<double> tx_candidates;
    for (int i = 0; i < tx_search_steps; ++i) {
        double tx = super_cx - actual_local_search_tx_half_width + 
                   (2.0 * actual_local_search_tx_half_width * i) / (tx_search_steps - 1);
        tx_candidates.push_back(tx);
    }
    if (tx_search_steps == 1) {
        tx_candidates = {super_cx};
    }
    
    // ty_candidates 생성
    std::vector<double> ty_candidates;
    for (int i = 0; i < ty_search_steps; ++i) {
        double ty = super_cy - actual_local_search_ty_half_width + 
                   (2.0 * actual_local_search_ty_half_width * i) / (ty_search_steps - 1);
        ty_candidates.push_back(ty);
    }
    if (ty_search_steps == 1) {
        ty_candidates = {super_cy};
    }
    
    double cell_best_score = -1.0;
    double cell_best_tx = 0.0, cell_best_ty = 0.0, cell_best_theta_deg = 0.0;
    int iterations_in_cell = 0;
    
    // 3중 루프 탐색 - 파이썬과 동일한 구조
    for (double tx_candidate : tx_candidates) {
        for (double ty_candidate : ty_candidates) {
            for (size_t k_theta = 0; k_theta < theta_candidates_rad.size(); ++k_theta) {
                iterations_in_cell++;
                
                // 키포인트 변환 - apply_transform_to_keypoints_numba와 동일
                std::vector<Keypoint> transformed_scan_kps;
                transformed_scan_kps.reserve(live_scan_keypoints.size());
                
                double theta_rad = theta_candidates_rad[k_theta];
                double cos_theta = std::cos(theta_rad);
                double sin_theta = std::sin(theta_rad);
                
                for (const auto& kp : live_scan_keypoints) {
                    Keypoint transformed_kp;
                    transformed_kp.x = kp.x * cos_theta - kp.y * sin_theta + tx_candidate;
                    transformed_kp.y = kp.x * sin_theta + kp.y * cos_theta + ty_candidate;
                    transformed_scan_kps.push_back(transformed_kp);
                }
                
                if (transformed_scan_kps.empty()) continue;
                
                // 대응점 계산 - count_correspondences_kdtree와 동일
                int current_score = countCorrespondencesKDTree(
                    transformed_scan_kps, global_map_keypoints, actual_correspondence_dist_thresh
                );
                
                if (current_score > cell_best_score) {
                    cell_best_score = current_score;
                    cell_best_tx = tx_candidate;
                    cell_best_ty = ty_candidate;
                    cell_best_theta_deg = theta_candidates_deg[k_theta];
                }
            }
        }
    }
    
    return CandidateInfo(cell_best_score, cell_best_tx, cell_best_ty, 
                        cell_best_theta_deg, super_cx, super_cy);
}

/**
 * @brief select_diverse_candidates 함수의 C++ 구현 - 파이썬과 동일한 로직
 */
std::vector<CandidateInfo> selectDiverseCandidates(
    const std::vector<CandidateInfo>& candidates_info,
    int num_to_select,
    double separation_factor,
    double cell_size_x,
    double cell_size_y,
    const std::pair<double, double>& map_x_range,
    const std::pair<double, double>& map_y_range
) {
    // 빈 후보 리스트 처리 - 파이썬과 동일
    if (candidates_info.empty()) {
        return {};
    }
    
    // 유효한 후보들만 필터링 - 파이썬의 valid_candidates 로직과 동일
    std::vector<CandidateInfo> valid_candidates;
    for (const auto& candidate : candidates_info) {
        if (candidate.score > -std::numeric_limits<double>::infinity() && 
            std::isfinite(candidate.center_x) && std::isfinite(candidate.center_y)) {
            valid_candidates.push_back(candidate);
        }
    }
    
    if (valid_candidates.empty()) {
        return {};
    }
    
    // 점수 기준으로 정렬 - 파이썬의 sorted_candidates와 동일 (점수 내림차순)
    std::sort(valid_candidates.begin(), valid_candidates.end(),
        [](const CandidateInfo& a, const CandidateInfo& b) {
            if (std::abs(a.score - b.score) < 1e-9) {
                // 점수가 같으면 tx, ty 순으로 정렬 (파이썬의 (x[0], x[1], x[2]) 로직)
                if (std::abs(a.tx - b.tx) < 1e-9) {
                    return a.ty > b.ty;
                }
                return a.tx > b.tx;
            }
            return a.score > b.score;  // 내림차순
        });
    
    std::vector<CandidateInfo> selected_candidates_final;
    double min_separation_dist_x = separation_factor * cell_size_x;
    double min_separation_dist_y = separation_factor * cell_size_y;
    
    // 파이썬과 동일한 NMS 로직
    for (const auto& cand_data : valid_candidates) {
        if (selected_candidates_final.size() >= static_cast<size_t>(num_to_select)) {
            break;
        }
        
        double cand_cx = cand_data.center_x;
        double cand_cy = cand_data.center_y;
        
        bool is_far_enough = true;
        if (!std::isfinite(cand_cx) || !std::isfinite(cand_cy)) {
            continue;
        }
        
        // 기존 선택된 후보들과의 거리 체크
        for (const auto& sel_cand_data : selected_candidates_final) {
            double sel_cx = sel_cand_data.center_x;
            double sel_cy = sel_cand_data.center_y;
            
            if (!std::isfinite(sel_cx) || !std::isfinite(sel_cy)) {
                continue;
            }
            
            if (std::abs(cand_cx - sel_cx) < min_separation_dist_x &&
                std::abs(cand_cy - sel_cy) < min_separation_dist_y) {
                is_far_enough = false;
                break;
            }
        }
        
        if (is_far_enough) {
            selected_candidates_final.push_back(cand_data);
        }
    }
    
    return selected_candidates_final;
}

/**
 * @brief search_single_level 함수의 C++ 구현 - 파이썬과 동일한 로직
 */
std::vector<CandidateInfo> searchSingleLevel(
    int level_idx,
    const std::vector<Keypoint>& global_map_keypoints,
    const std::vector<Keypoint>& live_scan_keypoints,
    const std::vector<double>& search_area_x_edges,
    const std::vector<double>& search_area_y_edges,
    const std::vector<int>& grid_division,
    const std::vector<double>& theta_candidates_rad,
    const std::vector<double>& theta_candidates_deg,
    double correspondence_dist_thresh,
    const std::vector<int>& tx_ty_search_steps_per_cell,
    double base_grid_cell_size,
    int num_processes,
    long long& total_iterations_evaluated_this_level
) {
    std::cout << "  Executing search_single_level for Level " << level_idx + 1 << "..." << std::endl;
    
    double map_x_min = search_area_x_edges.front();
    double map_x_max = search_area_x_edges.back();
    double map_y_min = search_area_y_edges.front();
    double map_y_max = search_area_y_edges.back();
    
    if (map_x_min >= map_x_max || map_y_min >= map_y_max) {
        std::cout << "    Warning: Invalid search area for Level " << level_idx + 1 
                  << ". Min/Max: X(" << std::fixed << std::setprecision(2) << map_x_min 
                  << "," << map_x_max << "), Y(" << map_y_min << "," << map_y_max 
                  << "). Skipping search." << std::endl;
        return {};
    }
    
    int num_super_x = grid_division[0];
    int num_super_y = grid_division[1];
    
    // super_x_edges 생성 - 파이썬의 np.linspace와 동일
    std::vector<double> super_x_edges;
    for (int i = 0; i <= num_super_x; ++i) {
        double edge = map_x_min + (map_x_max - map_x_min) * i / num_super_x;
        super_x_edges.push_back(edge);
    }
    
    std::vector<double> super_y_edges;
    for (int i = 0; i <= num_super_y; ++i) {
        double edge = map_y_min + (map_y_max - map_y_min) * i / num_super_y;
        super_y_edges.push_back(edge);
    }
    
    // 예외 처리 - 파이썬과 동일
    if (num_super_x == 0) {
        super_x_edges = {map_x_min, map_x_max};
        num_super_x = 1;
    }
    if (num_super_y == 0) {
        super_y_edges = {map_y_min, map_y_max};
        num_super_y = 1;
    }
    
    // 셀 중심 계산 - 파이썬과 동일
    std::vector<double> super_cx_centers, super_cy_centers;
    
    if (num_super_x > 0) {
        for (int i = 0; i < num_super_x; ++i) {
            super_cx_centers.push_back((super_x_edges[i] + super_x_edges[i + 1]) / 2.0);
        }
    } else {
        super_cx_centers.push_back((map_x_min + map_x_max) / 2.0);
    }
    
    if (num_super_y > 0) {
        for (int i = 0; i < num_super_y; ++i) {
            super_cy_centers.push_back((super_y_edges[i] + super_y_edges[i + 1]) / 2.0);
        }
    } else {
        super_cy_centers.push_back((map_y_min + map_y_max) / 2.0);
    }
    
    double cell_width = (num_super_x > 0) ? (map_x_max - map_x_min) / num_super_x : (map_x_max - map_x_min);
    double cell_height = (num_super_y > 0) ? (map_y_max - map_y_min) / num_super_y : (map_y_max - map_y_min);
    
    double actual_local_search_tx_half_width = cell_width / 2.0;
    double actual_local_search_ty_half_width = cell_height / 2.0;
    
    // 태스크 생성 - 파이썬과 동일
    std::vector<std::pair<double, double>> tasks;
    if (live_scan_keypoints.empty()) {
        std::cout << "    Warning: Live scan keypoints are empty. Skipping task generation in search_single_level." << std::endl;
    } else {
        for (double super_cx : super_cx_centers) {
            for (double super_cy : super_cy_centers) {
                tasks.emplace_back(super_cx, super_cy);
            }
        }
    }
    
    int total_super_cells = tasks.size();
    if (total_super_cells == 0) {
        return {};
    }
    
    std::vector<CandidateInfo> level_results;
    total_iterations_evaluated_this_level = 0;
    double current_level_best_score = -1.0;
    
    // 파이썬과 동일한 처리 로직
    if (num_processes == 0) {
        // Sequential processing - 파이썬과 동일
        for (const auto& task : tasks) {
            CandidateInfo result = processSuperGridCell(
                task.first, task.second, live_scan_keypoints, global_map_keypoints,
                actual_local_search_tx_half_width, actual_local_search_ty_half_width,
                tx_ty_search_steps_per_cell[0], tx_ty_search_steps_per_cell[1],
                theta_candidates_rad, theta_candidates_deg, correspondence_dist_thresh
            );
            
            if (result.score >= 0) {  // 유효한 점수인 경우
                level_results.push_back(result);
                if (result.score > current_level_best_score) {
                    current_level_best_score = result.score;
                }
            }
        }
    } else {
        // 멀티스레드 처리 - OpenMP 사용
        level_results.resize(tasks.size());
        
        #ifdef _OPENMP
        int actual_num_processes = (num_processes > 0) ? num_processes : omp_get_max_threads();
        omp_set_num_threads(actual_num_processes);
        
        #pragma omp parallel for schedule(dynamic)
        for (size_t i = 0; i < tasks.size(); ++i) {
            level_results[i] = processSuperGridCell(
                tasks[i].first, tasks[i].second, live_scan_keypoints, global_map_keypoints,
                actual_local_search_tx_half_width, actual_local_search_ty_half_width,
                tx_ty_search_steps_per_cell[0], tx_ty_search_steps_per_cell[1],
                theta_candidates_rad, theta_candidates_deg, correspondence_dist_thresh
            );
        }
        #else
        // OpenMP가 없는 경우 sequential로 처리
        for (size_t i = 0; i < tasks.size(); ++i) {
            level_results[i] = processSuperGridCell(
                tasks[i].first, tasks[i].second, live_scan_keypoints, global_map_keypoints,
                actual_local_search_tx_half_width, actual_local_search_ty_half_width,
                tx_ty_search_steps_per_cell[0], tx_ty_search_steps_per_cell[1],
                theta_candidates_rad, theta_candidates_deg, correspondence_dist_thresh
            );
        }
        #endif
        
        // 유효한 결과만 필터링
        std::vector<CandidateInfo> valid_results;
        for (const auto& result : level_results) {
            if (result.score >= 0) {
                valid_results.push_back(result);
                if (result.score > current_level_best_score) {
                    current_level_best_score = result.score;
                }
            }
        }
        level_results = valid_results;
    }
    
    return level_results;
}

/**
 * @brief hierarchical_adaptive_search 함수의 C++ 구현 - 파이썬과 완전히 동일한 로직
 */
TransformResult hierarchicalAdaptiveSearch(
    const std::vector<Keypoint>& global_map_keypoints,
    const std::vector<Keypoint>& live_scan_keypoints,
    const std::vector<double>& initial_map_x_edges,
    const std::vector<double>& initial_map_y_edges,
    const HierarchicalSearchParams& params
) {
    auto total_time_start = std::chrono::high_resolution_clock::now();
    
    std::cout << "🚀 Starting REAL C++ hierarchical adaptive search (Fully Implemented)" << std::endl;
    
    #ifdef _OPENMP
    int num_threads = (params.num_processes > 0) ? params.num_processes : omp_get_max_threads();
    std::cout << "🚀 Using " << num_threads << " OpenMP threads" << std::endl;
    #endif
    
    std::cout << "🚀 Processing " << params.level_configs.size()
              << " levels with " << global_map_keypoints.size()
              << " global keypoints and " << live_scan_keypoints.size()
              << " scan keypoints" << std::endl;
    
    // 입력 데이터 유효성 검증
    if (!validateInputData(global_map_keypoints, live_scan_keypoints, 
                          initial_map_x_edges, initial_map_y_edges, params)) {
        throw std::invalid_argument("Invalid input data for hierarchical search");
    }
    
    // 파이썬과 동일한 초기화
    double overall_best_score = -1.0;
    TransformResult overall_best_transform;
    overall_best_transform.tx = 0.0;
    overall_best_transform.ty = 0.0;
    overall_best_transform.theta_deg = 0.0;
    overall_best_transform.score = -1.0;
    
    long long grand_total_iterations_evaluated = 0;
    
    double initial_center_x = (initial_map_x_edges.front() + initial_map_x_edges.back()) / 2.0;
    double initial_center_y = (initial_map_y_edges.front() + initial_map_y_edges.back()) / 2.0;
    
    // processing_candidates_from_prev_level - 파이썬과 동일한 구조
    std::vector<CandidateInfo> processing_candidates_from_prev_level;
    processing_candidates_from_prev_level.emplace_back(
        -1.0, initial_center_x, initial_center_y, 0.0, initial_center_x, initial_center_y
    );
    
    // 레벨별 처리 - 파이썬과 동일한 로직
    for (size_t level_idx = 0; level_idx < params.level_configs.size(); ++level_idx) {
        const auto& config = params.level_configs[level_idx];
        
        std::cout << "\n=== Processing Level " << level_idx + 1 << " / " 
                  << params.level_configs.size() << " ===" << std::endl;
        
        std::vector<CandidateInfo> all_cell_infos_for_current_level_nms;
        
        if (processing_candidates_from_prev_level.empty()) {
            std::cout << "  Level " << level_idx + 1 
                      << ": No candidates from previous level to process. Stopping." << std::endl;
            break;
        }
        
        int num_prev_level_candidates = processing_candidates_from_prev_level.size();
        std::cout << "  Level " << level_idx + 1 << ": To explore " 
                  << num_prev_level_candidates << " candidate region(s) from previous level." << std::endl;
        
        // 각 후보 영역 탐색 - 파이썬과 동일
        for (size_t cand_idx = 0; cand_idx < processing_candidates_from_prev_level.size(); ++cand_idx) {
            const auto& prev_level_cand_info = processing_candidates_from_prev_level[cand_idx];
            
            double prev_score = prev_level_cand_info.score;
            double prev_tx = prev_level_cand_info.tx;
            double prev_ty = prev_level_cand_info.ty;
            double prev_theta = prev_level_cand_info.theta_deg;
            double prev_cx = prev_level_cand_info.center_x;
            double prev_cy = prev_level_cand_info.center_y;
            
            std::cout << "  Exploring candidate region " << cand_idx + 1 << "/" 
                      << num_prev_level_candidates << " (based on prev T: tx=" 
                      << std::fixed << std::setprecision(2) << prev_tx << ", ty=" << prev_ty 
                      << ", th=" << std::setprecision(1) << prev_theta << ")" << std::endl;
            
            // 탐색 영역 계산 - 파이썬과 동일한 로직
            std::vector<double> current_search_x_edges, current_search_y_edges;
            double center_for_theta_calc = 0.0;
            
            if (level_idx == 0) {
                current_search_x_edges = initial_map_x_edges;
                current_search_y_edges = initial_map_y_edges;
                center_for_theta_calc = 0.0;
            } else {
                double center_tx_for_current_search = prev_tx;
                double center_ty_for_current_search = prev_ty;
                center_for_theta_calc = prev_theta;
                
                double search_w = 20.0, search_h = 20.0;  // 기본값
                
                if (config.search_area_type == "relative_to_map" && !config.area_ratio_or_size.empty()) {
                    double map_width_init = initial_map_x_edges.back() - initial_map_x_edges.front();
                    double map_height_init = initial_map_y_edges.back() - initial_map_y_edges.front();
                    double ratio = config.area_ratio_or_size[0];
                    search_w = map_width_init * std::sqrt(ratio);
                    search_h = map_height_init * std::sqrt(ratio);
                } else if (config.search_area_type == "absolute_size" && config.area_ratio_or_size.size() >= 2) {
                    search_w = config.area_ratio_or_size[0];
                    search_h = config.area_ratio_or_size[1];
                } else {
                    std::cout << "Warning: Invalid search_area_type or area_param for Level " 
                              << level_idx + 1 << ". Defaulting to small area around prev_tx, prev_ty." << std::endl;
                }
                
                current_search_x_edges = {
                    center_tx_for_current_search - search_w / 2,
                    center_tx_for_current_search + search_w / 2
                };
                current_search_y_edges = {
                    center_ty_for_current_search - search_h / 2,
                    center_ty_for_current_search + search_h / 2
                };
                
                // 맵 경계 클리핑 - 파이썬과 동일
                current_search_x_edges[0] = std::max(current_search_x_edges[0], initial_map_x_edges.front());
                current_search_x_edges[0] = std::min(current_search_x_edges[0], initial_map_x_edges.back() - 1e-3);
                current_search_x_edges[1] = std::max(current_search_x_edges[1], initial_map_x_edges.front() + 1e-3);
                current_search_x_edges[1] = std::min(current_search_x_edges[1], initial_map_x_edges.back());
                
                current_search_y_edges[0] = std::max(current_search_y_edges[0], initial_map_y_edges.front());
                current_search_y_edges[0] = std::min(current_search_y_edges[0], initial_map_y_edges.back() - 1e-3);
                current_search_y_edges[1] = std::max(current_search_y_edges[1], initial_map_y_edges.front() + 1e-3);
                current_search_y_edges[1] = std::min(current_search_y_edges[1], initial_map_y_edges.back());
                
                if (current_search_x_edges[0] >= current_search_x_edges[1] || 
                    current_search_y_edges[0] >= current_search_y_edges[1]) {
                    std::cout << "    Skipping candidate region " << cand_idx + 1 
                              << " due to invalid search area after clipping." << std::endl;
                    continue;
                }
            }
            
            // theta 범위 계산 - 파이썬과 동일
            int theta_steps = config.theta_search_steps;
            double theta_min_deg, theta_max_deg;
            
            if (level_idx == 0 || config.search_area_type == "full_map") {
                theta_min_deg = config.theta_range_deg[0];
                theta_max_deg = config.theta_range_deg[1];
            } else {
                double theta_center_current = center_for_theta_calc;
                theta_min_deg = theta_center_current + config.theta_range_deg_relative[0];
                theta_max_deg = theta_center_current + config.theta_range_deg_relative[1];
            }
            
            // theta_candidates 생성 - 파이썬의 np.linspace와 동일
            std::vector<double> theta_candidates_deg, theta_candidates_rad;
            for (int i = 0; i < theta_steps; ++i) {
                double theta_deg = theta_min_deg + (theta_max_deg - theta_min_deg) * i / theta_steps;
                theta_candidates_deg.push_back(theta_deg);
                theta_candidates_rad.push_back(theta_deg * M_PI / 180.0);
            }
            
            // 단일 레벨 탐색 실행
            long long iterations_current_call = 0;
            std::vector<CandidateInfo> cell_infos_one_search = searchSingleLevel(
                level_idx, global_map_keypoints, live_scan_keypoints,
                current_search_x_edges, current_search_y_edges, config.grid_division,
                theta_candidates_rad, theta_candidates_deg,
                config.correspondence_dist_thresh_factor * params.base_grid_cell_size,
                config.tx_ty_search_steps_per_cell, params.base_grid_cell_size,
                params.num_processes, iterations_current_call
            );
            
            grand_total_iterations_evaluated += iterations_current_call;
            
            if (!cell_infos_one_search.empty()) {
                all_cell_infos_for_current_level_nms.insert(
                    all_cell_infos_for_current_level_nms.end(),
                    cell_infos_one_search.begin(), cell_infos_one_search.end()
                );
            } else {
                std::cout << "    No results from search_single_level for candidate region " 
                          << cand_idx + 1 << "." << std::endl;
            }
        }
        
        if (all_cell_infos_for_current_level_nms.empty()) {
            std::cout << "  Level " << level_idx + 1 
                      << ": No valid cell information obtained from any candidate region. Stopping." << std::endl;
            break;
        }
        
        // 전체 최고 점수 업데이트 - 파이썬과 동일
        for (const auto& info : all_cell_infos_for_current_level_nms) {
            if (info.score > overall_best_score) {
                overall_best_score = info.score;
                overall_best_transform.score = info.score;
                overall_best_transform.tx = info.tx;
                overall_best_transform.ty = info.ty;
                overall_best_transform.theta_deg = info.theta_deg;
                
                std::cout << "    New overall best (during Level " << level_idx + 1 
                          << " processing): score=" << std::fixed << std::setprecision(0) << overall_best_score 
                          << ", tx=" << std::setprecision(2) << info.tx << ", ty=" << info.ty 
                          << ", th=" << std::setprecision(1) << info.theta_deg << std::endl;
            }
        }
        
        // NMS 셀 크기 계산 - 파이썬과 동일
        double nms_cell_size_x, nms_cell_size_y;
        if (!all_cell_infos_for_current_level_nms.empty()) {
            // 첫 번째 탐색 영역을 기준으로 계산 (파이썬 로직)
            double first_search_width = initial_map_x_edges.back() - initial_map_x_edges.front();
            double first_search_height = initial_map_y_edges.back() - initial_map_y_edges.front();
            
            nms_cell_size_x = (config.grid_division[0] > 0) ? 
                first_search_width / config.grid_division[0] : first_search_width;
            nms_cell_size_y = (config.grid_division[1] > 0) ? 
                first_search_height / config.grid_division[1] : first_search_height;
        } else {
            nms_cell_size_x = (initial_map_x_edges.back() - initial_map_x_edges.front()) / 
                               std::max(1, config.grid_division[0]);
            nms_cell_size_y = (initial_map_y_edges.back() - initial_map_y_edges.front()) / 
                               std::max(1, config.grid_division[1]);
            std::cout << "Warning: NMS cell size calculation fallback due to no search bounds." << std::endl;
        }
        
        // 다음 레벨 후보 선택 - 파이썬과 동일
        int num_select_for_nms_this_level = (level_idx < params.level_configs.size() - 1) ? 
            params.num_candidates_to_select_per_level : 1;
        
        std::vector<CandidateInfo> selected_candidates_for_next_level = selectDiverseCandidates(
            all_cell_infos_for_current_level_nms,
            num_select_for_nms_this_level,
            params.min_candidate_separation_factor,
            nms_cell_size_x, nms_cell_size_y,
            {initial_map_x_edges.front(), initial_map_x_edges.back()},
            {initial_map_y_edges.front(), initial_map_y_edges.back()}
        );
        
        std::cout << "  Level " << level_idx + 1 << ": NMS selected " 
                  << selected_candidates_for_next_level.size() << " candidates for L" 
                  << level_idx + 2 << " (from " << all_cell_infos_for_current_level_nms.size() 
                  << " total cells)." << std::endl;
        
        if (selected_candidates_for_next_level.empty()) {
            std::cout << "  Level " << level_idx + 1 
                      << ": No candidates selected by NMS for further processing. Stopping." << std::endl;
            break;
        }
        
        processing_candidates_from_prev_level = selected_candidates_for_next_level;
        
        if (level_idx == params.level_configs.size() - 1) {
            std::cout << "\n--- Hierarchical search finished after " 
                      << params.level_configs.size() << " levels ---" << std::endl;
            break;
        }
    }
    
    // 최종 결과 설정
    if (!processing_candidates_from_prev_level.empty()) {
        const auto& final_best = processing_candidates_from_prev_level.front();
        overall_best_transform.score = final_best.score;
        overall_best_transform.tx = final_best.tx;
        overall_best_transform.ty = final_best.ty;
        overall_best_transform.theta_deg = final_best.theta_deg;
    }
    
    overall_best_transform.success = overall_best_transform.score >= 0;
    overall_best_transform.iterations = grand_total_iterations_evaluated;
    
    auto total_time_end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(total_time_end - total_time_start);
    overall_best_transform.execution_time_ms = duration.count();
    
    std::cout << "\n--- C++ Hierarchical Adaptive Search Complete ---" << std::endl;
    std::cout << "  Final Best Transform: tx=" << std::fixed << std::setprecision(2) 
              << overall_best_transform.tx << ", ty=" << overall_best_transform.ty
              << ", theta=" << std::setprecision(1) << overall_best_transform.theta_deg << " deg" << std::endl;
    std::cout << "  Best Score: " << std::setprecision(0) << overall_best_transform.score << std::endl;
    std::cout << "  Total Execution Time: " << overall_best_transform.execution_time_ms << " ms" << std::endl;
    
    return overall_best_transform;
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
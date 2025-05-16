#include "higgsr/registration.h"
#include <pcl/common/transforms.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <Eigen/Geometry>
#include <cmath>
#include <limits>
#include <algorithm>
#include <vector>
#include <chrono>
#include <iostream>

#if defined(_OPENMP)
#include <omp.h>
#endif

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

namespace higgsr {

// Helper to convert NumPy array to PCL PointCloud
pcl::PointCloud<pcl::PointXYZ>::Ptr numpyToPCL(py::array_t<float, py::array::c_style | py::array::forcecast>& np_array) {
    // Ensure np_array is N x 3
    if (np_array.ndim() != 2 || np_array.shape(1) != 3) {
        throw std::runtime_error("Input NumPy array must be N x 3. Received shape (" + std::to_string(np_array.shape(0)) + ", " + std::to_string(np_array.shape(1)) + ")");
    }
    auto rows = np_array.shape(0);
    
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>());
    cloud->width = static_cast<uint32_t>(rows);
    cloud->height = 1; // Unorganized point cloud
    cloud->points.resize(rows);

    auto buf = np_array.request();
    float* ptr = static_cast<float*>(buf.ptr);

    for (size_t i = 0; i < rows; ++i) {
        cloud->points[i].x = ptr[i * 3 + 0];
        cloud->points[i].y = ptr[i * 3 + 1];
        cloud->points[i].z = ptr[i * 3 + 2];
    }
    return cloud;
}

// Helper structure for candidate information (similar to Python tuple)
struct CandidateInfo {
    float score;
    float tx, ty, theta_deg;
    float cx, cy; // Super-cell center coordinates for NMS

    // 기본 생성자 추가
    CandidateInfo() : score(-1.0f), tx(0.f), ty(0.f), theta_deg(0.f), cx(0.f), cy(0.f) {}

    CandidateInfo(float s, float t_x, float t_y, float th, float c_x, float c_y)
        : score(s), tx(t_x), ty(t_y), theta_deg(th), cx(c_x), cy(c_y) {}

    // For sorting (higher score is better)
    bool operator<(const CandidateInfo& other) const {
        if (score != other.score) return score > other.score; // Primary sort by score (desc)
        if (tx != other.tx) return tx < other.tx;             // Secondary by tx
        return ty < other.ty;                                 // Tertiary by ty
    }
};

// Helper function to apply transform (placeholder for now)
pcl::PointCloud<pcl::PointXYZ>::Ptr apply_transform_to_keypoints_cpp(
    const pcl::PointCloud<pcl::PointXYZ>::Ptr& keypoints,
    float tx, float ty, float theta_rad) {
    
    Eigen::Affine3f transform = Eigen::Affine3f::Identity();
    transform.translation() << tx, ty, 0;
    transform.rotate(Eigen::AngleAxisf(theta_rad, Eigen::Vector3f::UnitZ()));
    
    pcl::PointCloud<pcl::PointXYZ>::Ptr transformed_keypoints(new pcl::PointCloud<pcl::PointXYZ>());
    pcl::transformPointCloud(*keypoints, *transformed_keypoints, transform);
    return transformed_keypoints;
}

// Helper function to count correspondences (placeholder for now)
int count_correspondences_kdtree_cpp(
    const pcl::PointCloud<pcl::PointXYZ>::Ptr& transformed_scan_keypoints,
    const pcl::KdTreeFLANN<pcl::PointXYZ>::Ptr& global_map_kdtree,
    float distance_threshold) {
    
    if (!transformed_scan_keypoints || transformed_scan_keypoints->empty() || !global_map_kdtree) {
        return 0;
    }
    int correspondence_count = 0;
    std::vector<int> point_idx_nkns(1);
    std::vector<float> point_nkns_sqr_dist(1);

    for (const auto& point : *transformed_scan_keypoints) {
        if (global_map_kdtree->nearestKSearch(point, 1, point_idx_nkns, point_nkns_sqr_dist) > 0) {
            if (std::sqrt(point_nkns_sqr_dist[0]) < distance_threshold) {
                correspondence_count++;
            }
        }
    }
    return correspondence_count;
}

// Placeholder for process_super_grid_cell
CandidateInfo process_super_grid_cell_cpp(
    float super_cx, float super_cy,
    const pcl::PointCloud<pcl::PointXYZ>::Ptr& live_scan_keypoints,
    const pcl::KdTreeFLANN<pcl::PointXYZ>::Ptr& global_map_kdtree,
    float actual_local_search_tx_half_width,
    float actual_local_search_ty_half_width,
    int tx_search_steps, int ty_search_steps,
    const std::vector<float>& theta_candidates_rad,
    const std::vector<float>& theta_candidates_deg, // For returning best theta in degrees
    float actual_correspondence_dist_thresh) {

    float cell_best_score = -1.0f;
    float cell_best_tx = 0.f, cell_best_ty = 0.f, cell_best_theta_deg = 0.f;

    std::vector<float> tx_candidates;
    for(int i = 0; i < tx_search_steps; ++i) {
        tx_candidates.push_back(super_cx - actual_local_search_tx_half_width + 
                                (2.0f * actual_local_search_tx_half_width * i) / (tx_search_steps -1));
    }
    if (tx_search_steps == 1) tx_candidates = {super_cx};


    std::vector<float> ty_candidates;
     for(int i = 0; i < ty_search_steps; ++i) {
        ty_candidates.push_back(super_cy - actual_local_search_ty_half_width +
                                (2.0f * actual_local_search_ty_half_width * i) / (ty_search_steps -1));
    }
    if (ty_search_steps == 1) ty_candidates = {super_cy};


    for (float tx_candidate : tx_candidates) {
        for (float ty_candidate : ty_candidates) {
            for (size_t k_theta = 0; k_theta < theta_candidates_rad.size(); ++k_theta) {
                pcl::PointCloud<pcl::PointXYZ>::Ptr transformed_kps = 
                    apply_transform_to_keypoints_cpp(live_scan_keypoints, tx_candidate, ty_candidate, theta_candidates_rad[k_theta]);
                
                if (transformed_kps->empty()) continue;

                int current_score = count_correspondences_kdtree_cpp(
                    transformed_kps, global_map_kdtree, actual_correspondence_dist_thresh
                );

                if (current_score > cell_best_score) {
                    cell_best_score = static_cast<float>(current_score);
                    cell_best_tx = tx_candidate;
                    cell_best_ty = ty_candidate;
                    cell_best_theta_deg = theta_candidates_deg[k_theta];
                }
            }
        }
    }
    return CandidateInfo(cell_best_score, cell_best_tx, cell_best_ty, cell_best_theta_deg, super_cx, super_cy);
}


// Placeholder for search_single_level
std::vector<CandidateInfo> search_single_level_cpp(
    int level_idx,
    const pcl::PointCloud<pcl::PointXYZ>::Ptr& global_map_keypoints,
    const pcl::PointCloud<pcl::PointXYZ>::Ptr& live_scan_keypoints,
    const std::vector<float>& search_area_x_edges, // {min_x, max_x}
    const std::vector<float>& search_area_y_edges, // {min_y, max_y}
    const std::vector<int>& grid_division, // {num_x, num_y}
    const std::vector<float>& theta_candidates_rad,
    const std::vector<float>& theta_candidates_deg,
    float correspondence_dist_thresh,
    const std::vector<int>& tx_ty_search_steps_per_cell, // {tx_steps, ty_steps}
    const pcl::KdTreeFLANN<pcl::PointXYZ>::Ptr& global_map_kdtree,
    int num_processes // For OpenMP
) {
    // std::cout << "  Executing search_single_level_cpp for Level " << level_idx + 1 << "..." << std::endl;
    float map_x_min = search_area_x_edges[0];
    float map_x_max = search_area_x_edges[1];
    float map_y_min = search_area_y_edges[0];
    float map_y_max = search_area_y_edges[1];

    if (map_x_min >= map_x_max || map_y_min >= map_y_max) {
        // std::cerr << "    Warning: Invalid search area for Level " << level_idx + 1 << ". Skipping search." << std::endl;
        return {};
    }

    int num_super_x = grid_division[0];
    int num_super_y = grid_division[1];

    std::vector<float> super_cx_centers;
    if (num_super_x > 0) {
        float step_x = (map_x_max - map_x_min) / num_super_x;
        for (int i = 0; i < num_super_x; ++i) super_cx_centers.push_back(map_x_min + step_x * (i + 0.5f));
    } else {
        super_cx_centers.push_back((map_x_min + map_x_max) / 2.0f);
        num_super_x = 1; // To avoid division by zero later
    }

    std::vector<float> super_cy_centers;
    if (num_super_y > 0) {
        float step_y = (map_y_max - map_y_min) / num_super_y;
        for (int i = 0; i < num_super_y; ++i) super_cy_centers.push_back(map_y_min + step_y * (i + 0.5f));
    } else {
        super_cy_centers.push_back((map_y_min + map_y_max) / 2.0f);
        num_super_y = 1; // To avoid division by zero later
    }
    
    float cell_width = (map_x_max - map_x_min) / num_super_x;
    float cell_height = (map_y_max - map_y_min) / num_super_y;
    float actual_local_search_tx_half_width = cell_width / 2.0f;
    float actual_local_search_ty_half_width = cell_height / 2.0f;

    std::vector<CandidateInfo> level_results;
    level_results.reserve(super_cx_centers.size() * super_cy_centers.size());

    if (live_scan_keypoints->empty()) {
         // std::cerr << "    Warning: Live scan keypoints are empty in search_single_level_cpp." << std::endl;
         return {};
    }

    std::vector<std::pair<float, float>> task_centers;
    for (float scx : super_cx_centers) {
        for (float scy : super_cy_centers) {
            task_centers.push_back({scx, scy});
        }
    }
    
    int actual_num_threads = 1;
    if (num_processes == 0) { // Sequential
        actual_num_threads = 1;
    } else if (num_processes == -1) { // Auto
        #if defined(_OPENMP)
        actual_num_threads = omp_get_max_threads();
        #endif
    } else { // User defined
        actual_num_threads = num_processes;
    }
    
    #if defined(_OPENMP)
    omp_set_num_threads(actual_num_threads);
    #endif

    std::vector<CandidateInfo> collected_results(task_centers.size());

    #pragma omp parallel for if(actual_num_threads > 1)
    for (size_t i = 0; i < task_centers.size(); ++i) {
        float super_cx = task_centers[i].first;
        float super_cy = task_centers[i].second;
        collected_results[i] = process_super_grid_cell_cpp(
            super_cx, super_cy,
            live_scan_keypoints, global_map_kdtree,
            actual_local_search_tx_half_width, actual_local_search_ty_half_width,
            tx_ty_search_steps_per_cell[0], tx_ty_search_steps_per_cell[1],
            theta_candidates_rad, theta_candidates_deg,
            correspondence_dist_thresh
        );
    }
    
    for(const auto& res : collected_results){
        if(res.score > -0.5f) { // Check if score is valid (not initial -1)
             level_results.push_back(res);
        }
    }
    return level_results;
}


// Placeholder for select_diverse_candidates_cpp
std::vector<CandidateInfo> select_diverse_candidates_cpp(
    std::vector<CandidateInfo>& candidates_info, // Pass by value to sort, or sort inside
    int num_to_select,
    float separation_factor,
    float cell_size_x, float cell_size_y) { // cell_size for this level's grid

    if (candidates_info.empty()) return {};

    // Filter out invalid candidates (score <= -0.5, assuming -1 is uninitialized)
    candidates_info.erase(std::remove_if(candidates_info.begin(), candidates_info.end(),
                                       [](const CandidateInfo& c){ return c.score < -0.5f || !std::isfinite(c.cx) || !std::isfinite(c.cy); }),
                        candidates_info.end());

    if (candidates_info.empty()) return {};
    
    std::sort(candidates_info.begin(), candidates_info.end()); // Uses operator< in CandidateInfo

    std::vector<CandidateInfo> selected_candidates_final;
    float min_separation_dist_x = separation_factor * cell_size_x;
    float min_separation_dist_y = separation_factor * cell_size_y;

    for (const auto& cand_data : candidates_info) {
        if (selected_candidates_final.size() >= static_cast<size_t>(num_to_select)) {
            break;
        }
        
        bool is_far_enough = true;
        if (!std::isfinite(cand_data.cx) || !std::isfinite(cand_data.cy)) {
            continue; 
        }

        for (const auto& sel_cand_data : selected_candidates_final) {
             if (!std::isfinite(sel_cand_data.cx) || !std::isfinite(sel_cand_data.cy)) {
                 continue; 
             }
            if (std::abs(cand_data.cx - sel_cand_data.cx) < min_separation_dist_x &&
                std::abs(cand_data.cy - sel_cand_data.cy) < min_separation_dist_y) {
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


Registration::Registration() {
    // 기본 설정 초기화
    config_.num_processes = 0; // Default to sequential, can be overridden by hierarchicalAdaptiveSearch
    config_.grid_cell_size = 0.2f;
    config_.correspondence_distance_threshold_factor = 2.5f;
}

Registration::~Registration() {
    // 필요한 정리 작업
}

// Modified hierarchicalAdaptiveSearch
RegistrationResult Registration::hierarchicalAdaptiveSearch(
    const pcl::PointCloud<pcl::PointXYZ>::Ptr& global_map_keypoints,
    const pcl::PointCloud<pcl::PointXYZ>::Ptr& live_scan_keypoints,
    const std::vector<float>& initial_map_x_edges,
    const std::vector<float>& initial_map_y_edges,
    const std::vector<LevelConfig>& level_configs,
    int num_candidates_to_select_per_level,
    float min_candidate_separation_factor,
    float base_grid_cell_size,
    int num_processes_param
) {
    RegistrationResult overall_best_result; // Automatically initializes with Identity and score -1
    // overall_best_result.transform = Eigen::Matrix4f::Identity(); // Redundant
    // overall_best_result.score = -1.0f; // Redundant

    if (!global_map_keypoints || global_map_keypoints->empty() || 
        !live_scan_keypoints || live_scan_keypoints->empty()) {
        std::cerr << "Error: Keypoints are empty for hierarchical search." << std::endl;
        return overall_best_result; // Return default (score -1)
    }
     if (initial_map_x_edges.size() != 2 || initial_map_y_edges.size() != 2 ||
        initial_map_x_edges[0] >= initial_map_x_edges[1] ||
        initial_map_y_edges[0] >= initial_map_y_edges[1]) {
        std::cerr << "Error: Invalid initial map edges." << std::endl;
        return overall_best_result;
    }
    if (level_configs.empty()){
        std::cerr << "Error: Level configurations are empty." << std::endl;
        return overall_best_result;
    }

    auto total_time_start = std::chrono::high_resolution_clock::now();
    long long grand_total_iterations_evaluated = 0; // Not fully implemented for iteration counting yet

    // KD-tree for global map
    pcl::KdTreeFLANN<pcl::PointXYZ>::Ptr global_map_kdtree(new pcl::KdTreeFLANN<pcl::PointXYZ>());
    global_map_kdtree->setInputCloud(global_map_keypoints);

    float initial_center_x = (initial_map_x_edges[0] + initial_map_x_edges[1]) / 2.0f;
    float initial_center_y = (initial_map_y_edges[0] + initial_map_y_edges[1]) / 2.0f;
    
    std::vector<CandidateInfo> processing_candidates_from_prev_level;
    // Initial candidate: score, tx, ty, theta, cell_center_x, cell_center_y
    processing_candidates_from_prev_level.emplace_back(-1.0f, initial_center_x, initial_center_y, 0.0f, initial_center_x, initial_center_y);

    for (size_t level_idx = 0; level_idx < level_configs.size(); ++level_idx) {
        const auto& config = level_configs[level_idx];
        // std::cout << "=== Processing Level " << level_idx + 1 << " / " << level_configs.size() << " ===" << std::endl;
        
        std::vector<CandidateInfo> all_cell_infos_for_current_level_nms;

        if (processing_candidates_from_prev_level.empty()) {
            // std::cout << "  Level " << level_idx + 1 << ": No candidates from previous level. Stopping." << std::endl;
            break;
        }
        
        // std::cout << "  Level " << level_idx + 1 << ": To explore " << processing_candidates_from_prev_level.size() 
        //           << " candidate region(s) from previous level." << std::endl;

        // current_search_x_edges 와 current_search_y_edges를 루프 시작 시 선언
        std::vector<float> current_search_x_edges(2);
        std::vector<float> current_search_y_edges(2);

        for (size_t cand_idx = 0; cand_idx < processing_candidates_from_prev_level.size(); ++cand_idx) {
            const auto& prev_level_cand_info = processing_candidates_from_prev_level[cand_idx];
            // std::cout << "  Exploring candidate region " << cand_idx + 1 << "/" << processing_candidates_from_prev_level.size()
            //           << " (based on prev T: tx=" << prev_level_cand_info.tx << ", ty=" << prev_level_cand_info.ty 
            //           << ", th=" << prev_level_cand_info.theta_deg << ")" << std::endl;

            float center_for_theta_calc_deg = 0.0f;

            if (level_idx == 0) {
                current_search_x_edges = initial_map_x_edges;
                current_search_y_edges = initial_map_y_edges;
                // center_for_theta_calc_deg is already 0.0f
            } else {
                float center_tx_for_current_search = prev_level_cand_info.tx;
                float center_ty_for_current_search = prev_level_cand_info.ty;
                center_for_theta_calc_deg = prev_level_cand_info.theta_deg;

                float search_w, search_h;
                if (config.search_area_type == "relative_to_map" && config.area_param.size() == 1) {
                    float map_width_init = initial_map_x_edges[1] - initial_map_x_edges[0];
                    float map_height_init = initial_map_y_edges[1] - initial_map_y_edges[0];
                    float ratio = config.area_param[0];
                    search_w = map_width_init * std::sqrt(ratio);
                    search_h = map_height_init * std::sqrt(ratio);
                } else if (config.search_area_type == "absolute_size" && config.area_param.size() == 2) {
                    search_w = config.area_param[0];
                    search_h = config.area_param[1];
                } else if (config.search_area_type == "full_map") { // Added for completeness
                     current_search_x_edges = initial_map_x_edges;
                     current_search_y_edges = initial_map_y_edges;
                     search_w = 0; // Flag to skip recalculation
                }
                else {
                    // std::cerr << "Warning: Invalid search_area_type or area_param for Level " << level_idx + 1 
                    //           << ". Defaulting to small area." << std::endl;
                    search_w = 20.0f * base_grid_cell_size; // Example: 20 cells wide
                    search_h = 20.0f * base_grid_cell_size; // Example: 20 cells high
                }

                if (search_w > 0) { // if not full_map
                    current_search_x_edges[0] = center_tx_for_current_search - search_w / 2.0f;
                    current_search_x_edges[1] = center_tx_for_current_search + search_w / 2.0f;
                    current_search_y_edges[0] = center_ty_for_current_search - search_h / 2.0f;
                    current_search_y_edges[1] = center_ty_for_current_search + search_h / 2.0f;

                    // Clip to initial map boundaries
                    current_search_x_edges[0] = std::max(initial_map_x_edges[0], std::min(current_search_x_edges[0], initial_map_x_edges[1] - 1e-3f));
                    current_search_x_edges[1] = std::max(initial_map_x_edges[0] + 1e-3f, std::min(current_search_x_edges[1], initial_map_x_edges[1]));
                    current_search_y_edges[0] = std::max(initial_map_y_edges[0], std::min(current_search_y_edges[0], initial_map_y_edges[1] - 1e-3f));
                    current_search_y_edges[1] = std::max(initial_map_y_edges[0] + 1e-3f, std::min(current_search_y_edges[1], initial_map_y_edges[1]));
                
                    if (current_search_x_edges[0] >= current_search_x_edges[1] || current_search_y_edges[0] >= current_search_y_edges[1]) {
                        // std::cout << "    Skipping candidate region " << cand_idx + 1 << " due to invalid search area after clipping." << std::endl;
                        continue;
                    }
                }
            }
            
            std::vector<float> theta_candidates_deg_level;
            std::vector<float> theta_candidates_rad_level;
            float theta_min_deg, theta_max_deg;

            if (level_idx == 0 || config.search_area_type == "full_map") {
                if (config.theta_range_deg.size() != 2) { /* error handling */ return overall_best_result; }
                theta_min_deg = config.theta_range_deg[0];
                theta_max_deg = config.theta_range_deg[1];
            } else {
                if (config.theta_range_deg_relative.size() != 2) { /* error handling */ return overall_best_result; }
                theta_min_deg = center_for_theta_calc_deg + config.theta_range_deg_relative[0];
                theta_max_deg = center_for_theta_calc_deg + config.theta_range_deg_relative[1];
            }

            if (config.theta_search_steps <=0) { /* error handling */ return overall_best_result; }
            float theta_step = (config.theta_search_steps > 1) ? (theta_max_deg - theta_min_deg) / config.theta_search_steps : 0;
            for (int i = 0; i < config.theta_search_steps; ++i) {
                float current_theta_deg = theta_min_deg + i * theta_step;
                theta_candidates_deg_level.push_back(current_theta_deg);
                theta_candidates_rad_level.push_back(current_theta_deg * M_PI / 180.0f);
            }
            if (config.theta_search_steps == 1) { // Single theta, use the center or min
                 theta_candidates_deg_level = {theta_min_deg};
                 theta_candidates_rad_level = {theta_min_deg * static_cast<float>(M_PI) / 180.0f};
            }


            std::vector<CandidateInfo> cell_infos_one_search = search_single_level_cpp(
                level_idx, global_map_keypoints, live_scan_keypoints,
                current_search_x_edges, current_search_y_edges, config.grid_division,
                theta_candidates_rad_level, theta_candidates_deg_level,
                config.correspondence_distance_threshold_factor * base_grid_cell_size,
                config.tx_ty_search_steps_per_cell,
                global_map_kdtree,
                num_processes_param
            );
            // grand_total_iterations_evaluated += ... (need to get this from search_single_level_cpp)
            
            if (!cell_infos_one_search.empty()) {
                all_cell_infos_for_current_level_nms.insert(
                    all_cell_infos_for_current_level_nms.end(),
                    cell_infos_one_search.begin(),
                    cell_infos_one_search.end()
                );
            } else {
                 // std::cout << "    No results from search_single_level_cpp for candidate region " << cand_idx + 1 << "." << std::endl;
            }
        } // End loop over processing_candidates_from_prev_level

        if (all_cell_infos_for_current_level_nms.empty()) {
            // std::cout << "  Level " << level_idx + 1 << ": No valid cell information obtained. Stopping." << std::endl;
            break;
        }

        for (const auto& cand : all_cell_infos_for_current_level_nms) {
            if (cand.score > overall_best_result.score) {
                overall_best_result.score = cand.score;
                // Update overall_best_result.transform (tx, ty, theta_deg)
                Eigen::Affine3f best_affine = Eigen::Affine3f::Identity();
                best_affine.translation() << cand.tx, cand.ty, 0;
                best_affine.rotate(Eigen::AngleAxisf(cand.theta_deg * M_PI / 180.0f, Eigen::Vector3f::UnitZ()));
                overall_best_result.transform = best_affine.matrix();
                // std::cout << "    New overall best (Level " << level_idx + 1 << "): score=" << cand.score
                //           << ", tx=" << cand.tx << ", ty=" << cand.ty << ", th=" << cand.theta_deg << std::endl;
            }
        }
        
        // NMS logic
        float nms_cell_size_x = 0.f, nms_cell_size_y = 0.f;
        if (config.grid_division.size() == 2 && config.grid_division[0] > 0 && config.grid_division[1] > 0) { 
            // A more accurate calculation based on the first candidate's search area (if level > 0)
            // or initial map (if level == 0) and current level's grid_division:
            // std::vector<float> area_for_nms_x = (level_idx == 0) ? initial_map_x_edges : processing_candidates_from_prev_level[0].cx; // This logic for cx, cy is not right for area - 주석처리
            // ... (기존 코드) ...
        } else {
             // std::cerr << "Warning: NMS cell size calculation fallback." << std::endl;
             nms_cell_size_x = base_grid_cell_size * 5.0f; // Fallback
             nms_cell_size_y = base_grid_cell_size * 5.0f; // Fallback
        }


        int num_select_for_nms = (level_idx == level_configs.size() - 1) ? 1 : num_candidates_to_select_per_level;
        
        processing_candidates_from_prev_level = select_diverse_candidates_cpp(
            all_cell_infos_for_current_level_nms, // This modifies the vector (sorts it)
            num_select_for_nms,
            min_candidate_separation_factor,
            nms_cell_size_x, nms_cell_size_y
        );
        
        // std::cout << "  Level " << level_idx + 1 << ": NMS selected " << processing_candidates_from_prev_level.size() 
        //           << " candidates for L" << level_idx + 2 << " (from " << all_cell_infos_for_current_level_nms.size() << " total cells)." << std::endl;

        if (processing_candidates_from_prev_level.empty()) {
            // std::cout << "  Level " << level_idx + 1 << ": No candidates selected by NMS. Stopping." << std::endl;
            break;
        }
        
        // Update overall best from NMS selected candidates if any are better (should already be caught above)
        // This is more for ensuring the final reported transform comes from the last selected set if hierarchy completes.
        if (level_idx == level_configs.size() - 1 && !processing_candidates_from_prev_level.empty()) {
             const auto& final_cand = processing_candidates_from_prev_level[0]; // Best after last NMS
             if (final_cand.score > overall_best_result.score) {
                 overall_best_result.score = final_cand.score;
                 Eigen::Affine3f final_affine = Eigen::Affine3f::Identity();
                 final_affine.translation() << final_cand.tx, final_cand.ty, 0;
                 final_affine.rotate(Eigen::AngleAxisf(final_cand.theta_deg * M_PI / 180.0f, Eigen::Vector3f::UnitZ()));
                 overall_best_result.transform = final_affine.matrix();
             }
        }


        if (level_idx == level_configs.size() - 1) {
            std::cout << "--- Hierarchical search finished after " << level_configs.size() << " levels ---" << std::endl;
            break;
        }
    } // End main level loop

    auto total_time_end = std::chrono::high_resolution_clock::now();
    overall_best_result.time_elapsed_sec = std::chrono::duration<double>(total_time_end - total_time_start).count();
    overall_best_result.total_iterations = grand_total_iterations_evaluated; // Still need to implement this sum

    std::cout << "--- Hierarchical Adaptive Search Complete (C++) ---" << std::endl;
    if (!processing_candidates_from_prev_level.empty()) {
        const auto& final_best_candidate = processing_candidates_from_prev_level[0];
        // Ensure the final result reflects the absolute best found, or the best from the last NMS.
        // The current overall_best_result.score should be the global max.
        // If the best from the last NMS is different but also very good, it might be chosen by Python's logic.
        // Here, we ensure overall_best_result is truly the best.
        // If overall_best_result was updated correctly through all levels, this step might be redundant
        // unless the last NMS somehow filters out the true global best due to diversity.
        // Python code:
        // final_selected_transform = overall_best_transform
        // final_selected_score = overall_best_score
        // if processing_candidates_from_prev_level: (updates if last level NMS output is better or simply sets it)
        // Let's make sure overall_best_result is truly the best one.
         if (final_best_candidate.score > overall_best_result.score) {
            overall_best_result.score = final_best_candidate.score;
            Eigen::Affine3f transform = Eigen::Affine3f::Identity();
            transform.translation() << final_best_candidate.tx, final_best_candidate.ty, 0;
            transform.rotate(Eigen::AngleAxisf(final_best_candidate.theta_deg * M_PI / 180.0f, Eigen::Vector3f::UnitZ()));
            overall_best_result.transform = transform.matrix();
         }
        
        // std::cout << "  Final Best Transform (from last NMS or overall best): tx=" << Eigen::Affine3f(overall_best_result.transform).translation().x()
        //           << ", ty=" << Eigen::Affine3f(overall_best_result.transform).translation().y()
        //           << ", theta=" << Eigen::AngleAxisf(Eigen::Affine3f(overall_best_result.transform).rotation()).angle() * 180.0f / M_PI
        //           << " deg (Score: " << overall_best_result.score << ")" << std::endl;

    } else if (overall_best_result.score < -0.5f) { // if score is still initial -1
        // std::cout << "  No valid transformation found." << std::endl;
    } else {
         // std::cout << "  Overall Best Transform (no candidates from final NMS, or NMS was skipped): tx=" << Eigen::Affine3f(overall_best_result.transform).translation().x()
         //          << ", ty=" << Eigen::Affine3f(overall_best_result.transform).translation().y()
         //          << ", theta=" << Eigen::AngleAxisf(Eigen::Affine3f(overall_best_result.transform).rotation()).angle() * 180.0f / M_PI
         //          << " deg (Score: " << overall_best_result.score << ")" << std::endl;
    }
    
    // std::cout << "  Total C++ Search Time: " << overall_best_result.time_elapsed_sec << " s" << std::endl;
    // std::cout << "  Total Transformations Evaluated: " << overall_best_result.total_iterations << std::endl;

    return overall_best_result;
}

std::tuple<Eigen::MatrixXf, Eigen::VectorXf, Eigen::VectorXf> Registration::create2DHeightVarianceMap(
    const pcl::PointCloud<pcl::PointXYZ>::Ptr& points_3d,
    float grid_cell_size,
    int min_points_per_cell,
    const std::string& density_metric
) {
    // 구현 필요
    // 비어있는 맵과 경계를 반환
    // std::cout << "[C++] create2DHeightVarianceMap called (Not Implemented)" << std::endl;
    return std::make_tuple(Eigen::MatrixXf(0,0), Eigen::VectorXf(0), Eigen::VectorXf(0));
}

std::vector<Eigen::Vector2f> Registration::extractHighDensityKeypoints(
    const Eigen::MatrixXf& density_map,
    const Eigen::VectorXf& x_edges,
    const Eigen::VectorXf& y_edges,
    float density_threshold
) {
    std::vector<Eigen::Vector2f> keypoints;
    // std::cout << "[C++] extractHighDensityKeypoints called (Not Implemented)" << std::endl;
    // 구현 필요
    // 현재는 빈 키포인트 벡터를 반환
    return keypoints;
}

} // namespace higgsr 


// Python 바인딩 코드
PYBIND11_MODULE(HiGGSR_cpp, m) {
    m.doc() = "Python bindings for HiGGSR C++ library";

    py::class_<higgsr::LevelConfig>(m, "LevelConfig")
        .def(py::init<>())
        .def_readwrite("grid_division", &higgsr::LevelConfig::grid_division)
        .def_readwrite("search_area_type", &higgsr::LevelConfig::search_area_type)
        .def_readwrite("area_param", &higgsr::LevelConfig::area_param)
        .def_readwrite("theta_range_deg", &higgsr::LevelConfig::theta_range_deg)
        .def_readwrite("theta_range_deg_relative", &higgsr::LevelConfig::theta_range_deg_relative)
        .def_readwrite("theta_search_steps", &higgsr::LevelConfig::theta_search_steps)
        .def_readwrite("correspondence_distance_threshold_factor", &higgsr::LevelConfig::correspondence_distance_threshold_factor)
        .def_readwrite("tx_ty_search_steps_per_cell", &higgsr::LevelConfig::tx_ty_search_steps_per_cell)
        .def("__repr__", [](const higgsr::LevelConfig &lc) {
            return "<LevelConfig with grid_division " + py::str(py::cast(lc.grid_division)).cast<std::string>() + ">";
        });

    py::class_<higgsr::RegistrationResult>(m, "RegistrationResult")
        .def(py::init<>())
        .def_readonly("transform", &higgsr::RegistrationResult::transform)
        .def_readonly("score", &higgsr::RegistrationResult::score)
        .def_readonly("time_elapsed_sec", &higgsr::RegistrationResult::time_elapsed_sec)
        .def_readonly("total_iterations", &higgsr::RegistrationResult::total_iterations)
        .def("__repr__", [](const higgsr::RegistrationResult &rr) {
            return "<RegistrationResult with score " + std::to_string(rr.score) + ">";
        });

    py::class_<higgsr::Registration>(m, "Registration")
        .def(py::init<>())
        .def("hierarchical_adaptive_search", 
             [](higgsr::Registration &self,
                py::array_t<float, py::array::c_style | py::array::forcecast> global_map_keypoints_np,
                py::array_t<float, py::array::c_style | py::array::forcecast> live_scan_keypoints_np,
                const std::vector<float>& initial_map_x_edges,
                const std::vector<float>& initial_map_y_edges,
                const std::vector<higgsr::LevelConfig>& level_configs,
                int num_candidates_to_select_per_level,
                float min_candidate_separation_factor,
                float base_grid_cell_size = 0.2f,
                int num_processes = 0) {
                    
                 if (global_map_keypoints_np.size() == 0 || live_scan_keypoints_np.size() == 0) {
                     std::cerr << "Warning: Empty keypoints provided to C++ binding." << std::endl;
                     higgsr::RegistrationResult res; // 기본값 반환 (score -1)
                     return res;
                 }

                 pcl::PointCloud<pcl::PointXYZ>::Ptr global_map_pcl = higgsr::numpyToPCL(global_map_keypoints_np);
                 pcl::PointCloud<pcl::PointXYZ>::Ptr live_scan_pcl = higgsr::numpyToPCL(live_scan_keypoints_np);
                 
                 return self.hierarchicalAdaptiveSearch(
                     global_map_pcl, live_scan_pcl,
                     initial_map_x_edges, initial_map_y_edges,
                     level_configs, num_candidates_to_select_per_level,
                     min_candidate_separation_factor, base_grid_cell_size, num_processes
                 );
             },
             py::arg("global_map_keypoints_np"),
             py::arg("live_scan_keypoints_np"),
             py::arg("initial_map_x_edges"),
             py::arg("initial_map_y_edges"),
             py::arg("level_configs"),
             py::arg("num_candidates_to_select_per_level"),
             py::arg("min_candidate_separation_factor"),
             py::arg("base_grid_cell_size") = 0.2f,
             py::arg("num_processes") = 0,
             "Performs hierarchical adaptive global registration using NumPy arrays as input for point clouds."
        )
        .def("create_2d_height_variance_map", // Python 스타일 이름
             [](higgsr::Registration &self,
                py::array_t<float, py::array::c_style | py::array::forcecast> points_3d_np,
                float grid_cell_size,
                int min_points_per_cell = 3,
                const std::string& density_metric = "std") {
                 
                 pcl::PointCloud<pcl::PointXYZ>::Ptr points_3d_pcl = higgsr::numpyToPCL(points_3d_np);
                 return self.create2DHeightVarianceMap(points_3d_pcl, grid_cell_size, min_points_per_cell, density_metric);
             },
             py::arg("points_3d_np"),
             py::arg("grid_cell_size"),
             py::arg("min_points_per_cell") = 3,
             py::arg("density_metric") = "std"
        )
        .def("extract_high_density_keypoints", &higgsr::Registration::extractHighDensityKeypoints,
             py::arg("density_map"),
             py::arg("x_edges"),
             py::arg("y_edges"),
             py::arg("density_threshold")
        );
} 
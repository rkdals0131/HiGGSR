#include "higgs_rc/core/registration.hpp"
#include "higgs_rc/core/feature_extraction.hpp"
#include "higgs_rc/core/utils.hpp"
#include <algorithm>
#include <numeric>
#include <omp.h>
#include <chrono>
#include <iostream>

// KD-tree implementation using PCL
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>

namespace higgs_rc {
namespace core {

// KDTree implementation using PCL
class KDTree::Impl {
public:
    pcl::PointCloud<pcl::PointXY>::Ptr cloud;
    pcl::KdTreeFLANN<pcl::PointXY> tree;

    Impl(const Eigen::MatrixXf& pts) {
        cloud = pcl::PointCloud<pcl::PointXY>::Ptr(new pcl::PointCloud<pcl::PointXY>);
        cloud->resize(pts.rows());
        
        for (int i = 0; i < pts.rows(); ++i) {
            cloud->points[i].x = pts(i, 0);
            cloud->points[i].y = pts(i, 1);
        }
        
        tree.setInputCloud(cloud);
    }
};

KDTree::KDTree(const Eigen::MatrixXf& points) 
    : pImpl(std::make_unique<Impl>(points)) {}

KDTree::~KDTree() = default;

std::vector<std::pair<int, float>> KDTree::queryRadius(
    const Eigen::Vector2f& point, float radius) const {
    
    pcl::PointXY searchPoint;
    searchPoint.x = point(0);
    searchPoint.y = point(1);
    
    std::vector<int> indices;
    std::vector<float> distances;
    
    pImpl->tree.radiusSearch(searchPoint, radius, indices, distances);
    
    std::vector<std::pair<int, float>> results;
    for (size_t i = 0; i < indices.size(); ++i) {
        results.emplace_back(indices[i], std::sqrt(distances[i]));
    }
    
    return results;
}

std::pair<int, float> KDTree::queryNearest(const Eigen::Vector2f& point) const {
    pcl::PointXY searchPoint;
    searchPoint.x = point(0);
    searchPoint.y = point(1);
    
    std::vector<int> indices(1);
    std::vector<float> distances(1);
    
    pImpl->tree.nearestKSearch(searchPoint, 1, indices, distances);
    
    if (!indices.empty()) {
        return std::make_pair(indices[0], std::sqrt(distances[0]));
    }
    
    return std::make_pair(-1, std::numeric_limits<float>::max());
}

// Main registration functions
std::vector<CandidateInfo> selectDiverseCandidates(
    const std::vector<CandidateInfo>& candidates,
    int num_to_select,
    float separation_factor,
    float cell_size_x,
    float cell_size_y) {
    
    if (candidates.empty()) return {};
    
    // Filter valid candidates
    std::vector<CandidateInfo> valid_candidates;
    for (const auto& cand : candidates) {
        if (cand.score > -std::numeric_limits<float>::infinity()) {
            valid_candidates.push_back(cand);
        }
    }
    
    if (valid_candidates.empty()) return {};
    
    // Sort by score (descending)
    std::sort(valid_candidates.begin(), valid_candidates.end(),
        [](const CandidateInfo& a, const CandidateInfo& b) {
            return a.score > b.score;
        });
    
    std::vector<CandidateInfo> selected;
    float min_sep_x = separation_factor * cell_size_x;
    float min_sep_y = separation_factor * cell_size_y;
    
    for (const auto& cand : valid_candidates) {
        if (selected.size() >= num_to_select) break;
        
        bool is_far_enough = true;
        for (const auto& sel : selected) {
            if (std::abs(cand.center_x - sel.center_x) < min_sep_x &&
                std::abs(cand.center_y - sel.center_y) < min_sep_y) {
                is_far_enough = false;
                break;
            }
        }
        
        if (is_far_enough) {
            selected.push_back(cand);
        }
    }
    
    return selected;
}

int countCorrespondencesKDTree(
    const Eigen::MatrixXf& transformed_scan_keypoints,
    const std::shared_ptr<KDTree>& global_map_kdtree,
    float distance_threshold) {
    
    if (transformed_scan_keypoints.rows() == 0 || !global_map_kdtree) {
        return 0;
    }
    
    int count = 0;
    
    #pragma omp parallel for reduction(+:count)
    for (int i = 0; i < transformed_scan_keypoints.rows(); ++i) {
        auto result = global_map_kdtree->queryNearest(transformed_scan_keypoints.row(i));
        if (result.second <= distance_threshold) {
            count++;
        }
    }
    
    return count;
}

CandidateInfo processSuperGridCell(
    float super_cx,
    float super_cy,
    const Eigen::MatrixXf& live_scan_keypoints,
    const std::shared_ptr<KDTree>& global_map_kdtree,
    float local_search_tx_half_width,
    float local_search_ty_half_width,
    int tx_search_steps,
    int ty_search_steps,
    const std::vector<float>& theta_candidates_rad,
    const std::vector<float>& theta_candidates_deg,
    float correspondence_dist_thresh) {
    
    CandidateInfo best_candidate;
    best_candidate.score = -1;
    best_candidate.center_x = super_cx;
    best_candidate.center_y = super_cy;
    
    // Generate translation candidates
    Eigen::VectorXf tx_candidates = Eigen::VectorXf::LinSpaced(
        tx_search_steps,
        super_cx - local_search_tx_half_width,
        super_cx + local_search_tx_half_width
    );
    
    Eigen::VectorXf ty_candidates = Eigen::VectorXf::LinSpaced(
        ty_search_steps,
        super_cy - local_search_ty_half_width,
        super_cy + local_search_ty_half_width
    );
    
    // Pre-allocate transformed keypoints matrix
    Eigen::MatrixXf transformed_scan_kps(live_scan_keypoints.rows(), 2);
    
    // Search over all combinations
    for (int i = 0; i < tx_candidates.size(); ++i) {
        float tx = tx_candidates(i);
        
        for (int j = 0; j < ty_candidates.size(); ++j) {
            float ty = ty_candidates(j);
            
            for (size_t k = 0; k < theta_candidates_rad.size(); ++k) {
                float theta_rad = theta_candidates_rad[k];
                float theta_deg = theta_candidates_deg[k];
                
                // Apply transformation
                applyTransformToKeypointsSIMD(
                    live_scan_keypoints, tx, ty, theta_rad, transformed_scan_kps
                );
                
                // Count correspondences
                int score = countCorrespondencesKDTree(
                    transformed_scan_kps, global_map_kdtree, correspondence_dist_thresh
                );
                
                // Update best if necessary
                if (score > best_candidate.score) {
                    best_candidate.score = score;
                    best_candidate.tx = tx;
                    best_candidate.ty = ty;
                    best_candidate.theta_deg = theta_deg;
                }
            }
        }
    }
    
    return best_candidate;
}

std::tuple<std::vector<CandidateInfo>, int> searchSingleLevel(
    int level_idx,
    const Eigen::MatrixXf& global_map_keypoints,
    const Eigen::MatrixXf& live_scan_keypoints,
    const Eigen::VectorXf& search_area_x_edges,
    const Eigen::VectorXf& search_area_y_edges,
    const LevelConfig& config,
    float base_grid_cell_size,
    const std::shared_ptr<KDTree>& global_map_kdtree,
    int num_processes) {
    
    std::cout << "  Executing search_single_level for Level " << level_idx + 1 << "..." << std::endl;
    
    float map_x_min = search_area_x_edges(0);
    float map_x_max = search_area_x_edges(search_area_x_edges.size() - 1);
    float map_y_min = search_area_y_edges(0);
    float map_y_max = search_area_y_edges(search_area_y_edges.size() - 1);
    
    if (map_x_min >= map_x_max || map_y_min >= map_y_max) {
        return std::make_tuple(std::vector<CandidateInfo>(), 0);
    }
    
    // Grid setup
    int num_super_x = config.grid_division[0];
    int num_super_y = config.grid_division[1];
    
    if (num_super_x == 0) num_super_x = 1;
    if (num_super_y == 0) num_super_y = 1;
    
    Eigen::VectorXf super_x_edges = Eigen::VectorXf::LinSpaced(
        num_super_x + 1, map_x_min, map_x_max
    );
    Eigen::VectorXf super_y_edges = Eigen::VectorXf::LinSpaced(
        num_super_y + 1, map_y_min, map_y_max
    );
    
    // Cell centers
    Eigen::VectorXf super_cx_centers(num_super_x);
    Eigen::VectorXf super_cy_centers(num_super_y);
    
    for (int i = 0; i < num_super_x; ++i) {
        super_cx_centers(i) = (super_x_edges(i) + super_x_edges(i + 1)) / 2.0f;
    }
    for (int i = 0; i < num_super_y; ++i) {
        super_cy_centers(i) = (super_y_edges(i) + super_y_edges(i + 1)) / 2.0f;
    }
    
    float cell_width = (map_x_max - map_x_min) / num_super_x;
    float cell_height = (map_y_max - map_y_min) / num_super_y;
    
    float local_search_tx_half_width = cell_width / 2.0f;
    float local_search_ty_half_width = cell_height / 2.0f;
    
    // Generate theta candidates
    std::vector<float> theta_candidates_rad;
    std::vector<float> theta_candidates_deg;
    
    float theta_min = config.theta_range_deg[0];
    float theta_max = config.theta_range_deg[1];
    
    for (int i = 0; i < config.theta_search_steps; ++i) {
        float theta_deg = theta_min + i * (theta_max - theta_min) / (config.theta_search_steps - 1);
        theta_candidates_deg.push_back(theta_deg);
        theta_candidates_rad.push_back(deg2rad(theta_deg));
    }
    
    float correspondence_dist_thresh = config.correspondence_distance_threshold_factor * base_grid_cell_size;
    
    // Process all grid cells
    std::vector<CandidateInfo> level_results;
    int total_iterations = 0;
    
    if (num_processes == 0) {
        // Sequential processing
        for (float cx : super_cx_centers) {
            for (float cy : super_cy_centers) {
                CandidateInfo result = processSuperGridCell(
                    cx, cy, live_scan_keypoints, global_map_kdtree,
                    local_search_tx_half_width, local_search_ty_half_width,
                    config.tx_ty_search_steps_per_cell[0],
                    config.tx_ty_search_steps_per_cell[1],
                    theta_candidates_rad, theta_candidates_deg,
                    correspondence_dist_thresh
                );
                
                if (result.score > 0) {
                    level_results.push_back(result);
                }
                
                total_iterations += config.tx_ty_search_steps_per_cell[0] * 
                                   config.tx_ty_search_steps_per_cell[1] * 
                                   config.theta_search_steps;
            }
        }
    } else {
        // Parallel processing
        int actual_num_processes = (num_processes < 0) ? omp_get_max_threads() : num_processes;
        omp_set_num_threads(actual_num_processes);
        
        std::vector<CandidateInfo> temp_results;
        
        #pragma omp parallel
        {
            std::vector<CandidateInfo> local_results;
            
            #pragma omp for collapse(2) schedule(dynamic)
            for (int i = 0; i < super_cx_centers.size(); ++i) {
                for (int j = 0; j < super_cy_centers.size(); ++j) {
                    CandidateInfo result = processSuperGridCell(
                        super_cx_centers(i), super_cy_centers(j),
                        live_scan_keypoints, global_map_kdtree,
                        local_search_tx_half_width, local_search_ty_half_width,
                        config.tx_ty_search_steps_per_cell[0],
                        config.tx_ty_search_steps_per_cell[1],
                        theta_candidates_rad, theta_candidates_deg,
                        correspondence_dist_thresh
                    );
                    
                    if (result.score > 0) {
                        local_results.push_back(result);
                    }
                }
            }
            
            #pragma omp critical
            {
                temp_results.insert(temp_results.end(), 
                                   local_results.begin(), 
                                   local_results.end());
            }
        }
        
        level_results = std::move(temp_results);
        total_iterations = num_super_x * num_super_y * 
                          config.tx_ty_search_steps_per_cell[0] * 
                          config.tx_ty_search_steps_per_cell[1] * 
                          config.theta_search_steps;
    }
    
    return std::make_tuple(level_results, total_iterations);
}

TransformResult hierarchicalAdaptiveSearch(
    const Eigen::MatrixXf& global_map_keypoints,
    const Eigen::MatrixXf& live_scan_keypoints,
    const Eigen::VectorXf& initial_map_x_edges,
    const Eigen::VectorXf& initial_map_y_edges,
    const std::vector<LevelConfig>& level_configs,
    int num_candidates_to_select_per_level,
    float min_candidate_separation_factor,
    float base_grid_cell_size,
    int num_processes) {
    
    TransformResult overall_best;
    
    if (global_map_keypoints.rows() == 0 || live_scan_keypoints.rows() == 0) {
        std::cerr << "Error: Keypoints are empty for hierarchical search." << std::endl;
        return overall_best;
    }
    
    // Build KD-tree for global map
    auto global_map_kdtree = std::make_shared<KDTree>(global_map_keypoints);
    
    // Initial candidates (center of the map)
    float initial_center_x = (initial_map_x_edges(0) + 
                             initial_map_x_edges(initial_map_x_edges.size() - 1)) / 2.0f;
    float initial_center_y = (initial_map_y_edges(0) + 
                             initial_map_y_edges(initial_map_y_edges.size() - 1)) / 2.0f;
    
    std::vector<CandidateInfo> processing_candidates;
    CandidateInfo initial_candidate;
    initial_candidate.score = -1;
    initial_candidate.tx = initial_center_x;
    initial_candidate.ty = initial_center_y;
    initial_candidate.theta_deg = 0;
    initial_candidate.center_x = initial_center_x;
    initial_candidate.center_y = initial_center_y;
    processing_candidates.push_back(initial_candidate);
    
    // Process each level
    for (size_t level_idx = 0; level_idx < level_configs.size(); ++level_idx) {
        std::cout << "\n=== Processing Level " << level_idx + 1 
                  << " / " << level_configs.size() << " ===" << std::endl;
        
        if (processing_candidates.empty()) {
            std::cout << "  No candidates from previous level. Stopping." << std::endl;
            break;
        }
        
        std::vector<CandidateInfo> all_candidates_this_level;
        
        // Process each candidate from previous level
        for (const auto& prev_candidate : processing_candidates) {
            // Determine search area based on config
            Eigen::VectorXf search_x_edges, search_y_edges;
            
            if (level_configs[level_idx].search_area_type == "full_map") {
                search_x_edges = initial_map_x_edges;
                search_y_edges = initial_map_y_edges;
            } else if (level_configs[level_idx].search_area_type == "relative_to_map") {
                float ratio = level_configs[level_idx].area_ratio_or_size;
                float map_width = initial_map_x_edges(initial_map_x_edges.size() - 1) - 
                                 initial_map_x_edges(0);
                float map_height = initial_map_y_edges(initial_map_y_edges.size() - 1) - 
                                  initial_map_y_edges(0);
                
                float search_width = map_width * ratio;
                float search_height = map_height * ratio;
                
                search_x_edges = Eigen::VectorXf::LinSpaced(
                    2,
                    prev_candidate.tx - search_width / 2,
                    prev_candidate.tx + search_width / 2
                );
                search_y_edges = Eigen::VectorXf::LinSpaced(
                    2,
                    prev_candidate.ty - search_height / 2,
                    prev_candidate.ty + search_height / 2
                );
            } else { // absolute_size
                float search_width = level_configs[level_idx].area_size[0];
                float search_height = level_configs[level_idx].area_size[1];
                
                search_x_edges = Eigen::VectorXf::LinSpaced(
                    2,
                    prev_candidate.tx - search_width / 2,
                    prev_candidate.tx + search_width / 2
                );
                search_y_edges = Eigen::VectorXf::LinSpaced(
                    2,
                    prev_candidate.ty - search_height / 2,
                    prev_candidate.ty + search_height / 2
                );
            }
            
            // Search this level
            auto [level_results, iterations] = searchSingleLevel(
                level_idx,
                global_map_keypoints,
                live_scan_keypoints,
                search_x_edges,
                search_y_edges,
                level_configs[level_idx],
                base_grid_cell_size,
                global_map_kdtree,
                num_processes
            );
            
            // Add to all candidates
            all_candidates_this_level.insert(
                all_candidates_this_level.end(),
                level_results.begin(),
                level_results.end()
            );
        }
        
        // Select diverse candidates for next level
        float cell_size_x = (initial_map_x_edges(initial_map_x_edges.size() - 1) - 
                            initial_map_x_edges(0)) / level_configs[level_idx].grid_division[0];
        float cell_size_y = (initial_map_y_edges(initial_map_y_edges.size() - 1) - 
                            initial_map_y_edges(0)) / level_configs[level_idx].grid_division[1];
        
        processing_candidates = selectDiverseCandidates(
            all_candidates_this_level,
            num_candidates_to_select_per_level,
            min_candidate_separation_factor,
            cell_size_x,
            cell_size_y
        );
        
        // Update overall best
        for (const auto& cand : all_candidates_this_level) {
            if (cand.score > overall_best.score) {
                overall_best.tx = cand.tx;
                overall_best.ty = cand.ty;
                overall_best.theta_deg = cand.theta_deg;
                overall_best.score = cand.score;
                overall_best.valid = true;
            }
        }
    }
    
    return overall_best;
}

} // namespace core
} // namespace higgs_rc
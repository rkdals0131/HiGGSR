#ifndef HIGGS_RC_CORE_REGISTRATION_HPP
#define HIGGS_RC_CORE_REGISTRATION_HPP

#include <Eigen/Core>
#include <vector>
#include <memory>
#include <tuple>

namespace higgs_rc {
namespace core {

// Forward declaration
class KDTree;

/**
 * @brief Level configuration for hierarchical search
 */
struct LevelConfig {
    std::array<int, 2> grid_division;
    std::string search_area_type;
    std::array<float, 2> theta_range_deg;
    int theta_search_steps;
    float correspondence_distance_threshold_factor;
    std::array<int, 2> tx_ty_search_steps_per_cell;
    
    // For relative_to_map or absolute_size types
    float area_ratio_or_size;
    std::array<float, 2> area_size;
    std::array<float, 2> theta_range_deg_relative;
};

/**
 * @brief Transform result structure
 */
struct TransformResult {
    float tx;
    float ty;
    float theta_deg;
    float score;
    bool valid;
    
    TransformResult() : tx(0), ty(0), theta_deg(0), score(-1), valid(false) {}
};

/**
 * @brief Candidate information for non-maximum suppression
 */
struct CandidateInfo {
    float score;
    float tx;
    float ty;
    float theta_deg;
    float center_x;
    float center_y;
};

/**
 * @brief Select diverse candidates using non-maximum suppression
 */
std::vector<CandidateInfo> selectDiverseCandidates(
    const std::vector<CandidateInfo>& candidates,
    int num_to_select,
    float separation_factor,
    float cell_size_x,
    float cell_size_y);

/**
 * @brief Count correspondences using KD-tree
 */
int countCorrespondencesKDTree(
    const Eigen::MatrixXf& transformed_scan_keypoints,
    const std::shared_ptr<KDTree>& global_map_kdtree,
    float distance_threshold);

/**
 * @brief Process a single super grid cell
 */
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
    float correspondence_dist_thresh);

/**
 * @brief Search in a single level of the hierarchy
 */
std::tuple<std::vector<CandidateInfo>, int> searchSingleLevel(
    int level_idx,
    const Eigen::MatrixXf& global_map_keypoints,
    const Eigen::MatrixXf& live_scan_keypoints,
    const Eigen::VectorXf& search_area_x_edges,
    const Eigen::VectorXf& search_area_y_edges,
    const LevelConfig& config,
    float base_grid_cell_size,
    const std::shared_ptr<KDTree>& global_map_kdtree,
    int num_processes);

/**
 * @brief Main hierarchical adaptive search algorithm
 */
TransformResult hierarchicalAdaptiveSearch(
    const Eigen::MatrixXf& global_map_keypoints,
    const Eigen::MatrixXf& live_scan_keypoints,
    const Eigen::VectorXf& initial_map_x_edges,
    const Eigen::VectorXf& initial_map_y_edges,
    const std::vector<LevelConfig>& level_configs,
    int num_candidates_to_select_per_level,
    float min_candidate_separation_factor,
    float base_grid_cell_size,
    int num_processes);

/**
 * @brief Simple KD-tree implementation for nearest neighbor search
 */
class KDTree {
public:
    KDTree(const Eigen::MatrixXf& points);
    ~KDTree();
    
    /**
     * @brief Query nearest neighbors within a distance threshold
     */
    std::vector<std::pair<int, float>> queryRadius(
        const Eigen::Vector2f& point,
        float radius) const;
    
    /**
     * @brief Query single nearest neighbor
     */
    std::pair<int, float> queryNearest(const Eigen::Vector2f& point) const;

private:
    class Impl;
    std::unique_ptr<Impl> pImpl;
};

} // namespace core
} // namespace higgs_rc

#endif // HIGGS_RC_CORE_REGISTRATION_HPP
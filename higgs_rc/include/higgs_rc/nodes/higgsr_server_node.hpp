#ifndef HIGGS_RC_NODES_HIGGSR_SERVER_NODE_HPP
#define HIGGS_RC_NODES_HIGGSR_SERVER_NODE_HPP

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <tf2_ros/transform_broadcaster.h>

#include "higgsr_interface/srv/register_scan.hpp"
#include "higgsr_interface/srv/set_global_map.hpp"
#include "higgs_rc/core/registration.hpp"

#include <memory>
#include <string>
#include <vector>
#include <mutex>

namespace higgs_rc {
namespace nodes {

class HiggsrServerNode : public rclcpp::Node {
public:
    explicit HiggsrServerNode(const rclcpp::NodeOptions& options = rclcpp::NodeOptions());
    ~HiggsrServerNode() = default;

private:
    // Services
    rclcpp::Service<higgsr_interface::srv::RegisterScan>::SharedPtr register_scan_service_;
    rclcpp::Service<higgsr_interface::srv::SetGlobalMap>::SharedPtr set_global_map_service_;
    
    // Publishers
    rclcpp::Publisher<geometry_msgs::msg::TransformStamped>::SharedPtr transform_pub_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr global_map_pub_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr live_scan_pub_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr global_keypoints_pub_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr scan_keypoints_pub_;
    
    // TF broadcaster
    std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
    
    // Core data
    Eigen::MatrixXf global_map_points_;
    Eigen::MatrixXf global_map_keypoints_;
    std::shared_ptr<core::KDTree> global_map_kdtree_;
    
    // Parameters
    std::string global_map_file_path_;
    float global_grid_size_;
    int global_min_points_for_density_calc_;
    std::string global_density_metric_;
    float global_keypoint_density_threshold_;
    std::string global_frame_id_;
    
    float live_grid_size_;
    int live_min_points_for_density_calc_;
    std::string live_density_metric_;
    float live_keypoint_density_threshold_;
    
    std::vector<core::LevelConfig> level_configs_;
    int num_candidates_per_level_;
    float min_candidate_separation_factor_;
    int num_processes_;
    
    bool enable_matplotlib_visualization_;
    bool enable_2d_keypoints_visualization_;
    bool enable_3d_result_visualization_;
    bool enable_super_grid_heatmap_visualization_;
    
    // Mutex for thread safety
    std::mutex registration_mutex_;
    
    // Service callbacks
    void registerScanCallback(
        const std::shared_ptr<higgsr_interface::srv::RegisterScan::Request> request,
        std::shared_ptr<higgsr_interface::srv::RegisterScan::Response> response);
    
    void setGlobalMapCallback(
        const std::shared_ptr<higgsr_interface::srv::SetGlobalMap::Request> request,
        std::shared_ptr<higgsr_interface::srv::SetGlobalMap::Response> response);
    
    // Helper functions
    bool loadGlobalMap(const std::string& filepath);
    void processGlobalMap();
    std::tuple<Eigen::MatrixXf, core::TransformResult> processLiveScan(
        const sensor_msgs::msg::PointCloud2& scan_msg);
    
    void publishVisualization(
        const Eigen::MatrixXf& live_scan_keypoints,
        const core::TransformResult& result);
    
    // Parameter parsing
    void parseParameters();
    std::vector<core::LevelConfig> parseLevelConfigs(const std::string& json_str);
};

} // namespace nodes
} // namespace higgs_rc

#endif // HIGGS_RC_NODES_HIGGSR_SERVER_NODE_HPP
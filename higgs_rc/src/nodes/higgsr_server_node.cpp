#include "higgs_rc/nodes/higgsr_server_node.hpp"
#include "higgs_rc/core/feature_extraction.hpp"
#include "higgs_rc/core/utils.hpp"

#include <sensor_msgs/point_cloud2_iterator.hpp>
#include <tf2_eigen/tf2_eigen.hpp>
#include <chrono>
#include <filesystem>
// #include <nlohmann/json.hpp>  // Optional for JSON parsing

namespace higgs_rc {
namespace nodes {

HiggsrServerNode::HiggsrServerNode(const rclcpp::NodeOptions& options)
    : Node("higgsr_server_node", options) {
    
    // Parse parameters
    parseParameters();
    
    // Create services
    register_scan_service_ = this->create_service<higgsr_interface::srv::RegisterScan>(
        "register_scan",
        std::bind(&HiggsrServerNode::registerScanCallback, this,
                  std::placeholders::_1, std::placeholders::_2));
    
    set_global_map_service_ = this->create_service<higgsr_interface::srv::SetGlobalMap>(
        "set_global_map",
        std::bind(&HiggsrServerNode::setGlobalMapCallback, this,
                  std::placeholders::_1, std::placeholders::_2));
    
    // Create publishers
    transform_pub_ = this->create_publisher<geometry_msgs::msg::TransformStamped>(
        "higgsr_transform", 10);
    global_map_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(
        "higgsr_global_map", rclcpp::QoS(1).transient_local());
    live_scan_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(
        "higgsr_live_scan", 10);
    global_keypoints_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>(
        "higgsr_global_keypoints", rclcpp::QoS(1).transient_local());
    scan_keypoints_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>(
        "higgsr_scan_keypoints", 10);
    
    // Create TF broadcaster
    tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);
    
    // Load global map on startup
    if (!global_map_file_path_.empty()) {
        RCLCPP_INFO(this->get_logger(), "Loading global map from: %s", 
                    global_map_file_path_.c_str());
        if (loadGlobalMap(global_map_file_path_)) {
            processGlobalMap();
            RCLCPP_INFO(this->get_logger(), "Global map loaded and processed successfully");
        } else {
            RCLCPP_ERROR(this->get_logger(), "Failed to load global map");
        }
    }
    
    RCLCPP_INFO(this->get_logger(), "HiGGSR Server Node initialized");
}

void HiggsrServerNode::parseParameters() {
    // Global map parameters
    global_map_file_path_ = this->declare_parameter<std::string>(
        "global_map.file_path", "");
    global_grid_size_ = this->declare_parameter<float>(
        "global_map.grid_size", 0.2);
    global_min_points_for_density_calc_ = this->declare_parameter<int>(
        "global_map.min_points_for_density", 3);
    global_density_metric_ = this->declare_parameter<std::string>(
        "global_map.density_metric", "std");
    global_keypoint_density_threshold_ = this->declare_parameter<float>(
        "global_map.keypoint_density_threshold", 0.1);
    global_frame_id_ = this->declare_parameter<std::string>(
        "global_map.frame_id", "map");
    
    // Live scan parameters
    live_grid_size_ = this->declare_parameter<float>(
        "live_scan.grid_size", 0.2);
    live_min_points_for_density_calc_ = this->declare_parameter<int>(
        "live_scan.min_points_for_density", 3);
    live_density_metric_ = this->declare_parameter<std::string>(
        "live_scan.density_metric", "std");
    live_keypoint_density_threshold_ = this->declare_parameter<float>(
        "live_scan.keypoint_density_threshold", 0.1);
    
    // Algorithm parameters
    num_candidates_per_level_ = this->declare_parameter<int>(
        "algorithm.num_candidates_per_level", 3);
    min_candidate_separation_factor_ = this->declare_parameter<float>(
        "algorithm.min_candidate_separation_factor", 1.5);
    num_processes_ = this->declare_parameter<int>(
        "algorithm.num_processes", 0);
    
    // Parse level configurations
    // Note: In production, you'd want to use a proper YAML parser for complex structures
    // For now, we'll use default values
    level_configs_.clear();
    
    // Level 1
    core::LevelConfig level1;
    level1.grid_division = {6, 6};
    level1.search_area_type = "full_map";
    level1.theta_range_deg = {0, 359};
    level1.theta_search_steps = 48;
    level1.correspondence_distance_threshold_factor = 7.0;
    level1.tx_ty_search_steps_per_cell = {10, 10};
    level_configs_.push_back(level1);
    
    // Level 2
    core::LevelConfig level2;
    level2.grid_division = {7, 7};
    level2.search_area_type = "relative_to_map";
    level2.area_ratio_or_size = 0.4;
    level2.theta_range_deg = {0, 359};
    level2.theta_search_steps = 48;
    level2.correspondence_distance_threshold_factor = 5.0;
    level2.tx_ty_search_steps_per_cell = {10, 10};
    level_configs_.push_back(level2);
    
    // Level 3
    core::LevelConfig level3;
    level3.grid_division = {4, 4};
    level3.search_area_type = "absolute_size";
    level3.area_size = {40.0, 40.0};
    level3.theta_range_deg = {0, 359};
    level3.theta_search_steps = 48;
    level3.correspondence_distance_threshold_factor = 2.5;
    level3.tx_ty_search_steps_per_cell = {10, 10};
    level_configs_.push_back(level3);
    
    // Visualization parameters
    enable_matplotlib_visualization_ = this->declare_parameter<bool>(
        "visualization.enable_matplotlib", true);
    enable_2d_keypoints_visualization_ = this->declare_parameter<bool>(
        "visualization.enable_2d_keypoints", true);
    enable_3d_result_visualization_ = this->declare_parameter<bool>(
        "visualization.enable_3d_result", true);
    enable_super_grid_heatmap_visualization_ = this->declare_parameter<bool>(
        "visualization.enable_super_grid_heatmap", true);
}

bool HiggsrServerNode::loadGlobalMap(const std::string& filepath) {
    try {
        // Check if file exists
        if (!std::filesystem::exists(filepath)) {
            RCLCPP_ERROR(this->get_logger(), "Global map file does not exist: %s", 
                        filepath.c_str());
            return false;
        }
        
        // Load point cloud
        global_map_points_ = core::loadPointCloudFromFile(filepath);
        
        if (global_map_points_.rows() == 0) {
            RCLCPP_ERROR(this->get_logger(), "Failed to load points from file");
            return false;
        }
        
        RCLCPP_INFO(this->get_logger(), "Loaded %ld points from global map", 
                    global_map_points_.rows());
        return true;
        
    } catch (const std::exception& e) {
        RCLCPP_ERROR(this->get_logger(), "Exception loading global map: %s", e.what());
        return false;
    }
}

void HiggsrServerNode::processGlobalMap() {
    std::lock_guard<std::mutex> lock(registration_mutex_);
    
    // Create density map
    auto [density_map, x_edges, y_edges] = core::create2DHeightVarianceMap(
        global_map_points_,
        global_grid_size_,
        global_min_points_for_density_calc_,
        global_density_metric_
    );
    
    // Extract keypoints
    global_map_keypoints_ = core::extractHighDensityKeypoints(
        density_map, x_edges, y_edges, global_keypoint_density_threshold_
    );
    
    // Build KD-tree
    if (global_map_keypoints_.rows() > 0) {
        global_map_kdtree_ = std::make_shared<core::KDTree>(global_map_keypoints_);
    }
    
    RCLCPP_INFO(this->get_logger(), 
                "Extracted %ld keypoints from global map", 
                global_map_keypoints_.rows());
    
    // TODO: Publish visualization markers
}

std::tuple<Eigen::MatrixXf, core::TransformResult> 
HiggsrServerNode::processLiveScan(const sensor_msgs::msg::PointCloud2& scan_msg) {
    // Convert PointCloud2 to Eigen matrix
    Eigen::MatrixXf scan_points(scan_msg.width * scan_msg.height, 3);
    
    sensor_msgs::PointCloud2ConstIterator<float> iter_x(scan_msg, "x");
    sensor_msgs::PointCloud2ConstIterator<float> iter_y(scan_msg, "y");
    sensor_msgs::PointCloud2ConstIterator<float> iter_z(scan_msg, "z");
    
    int idx = 0;
    for (; iter_x != iter_x.end(); ++iter_x, ++iter_y, ++iter_z) {
        scan_points(idx, 0) = *iter_x;
        scan_points(idx, 1) = *iter_y;
        scan_points(idx, 2) = *iter_z;
        idx++;
    }
    
    // Create density map
    auto [density_map, x_edges, y_edges] = core::create2DHeightVarianceMap(
        scan_points,
        live_grid_size_,
        live_min_points_for_density_calc_,
        live_density_metric_
    );
    
    // Extract keypoints
    Eigen::MatrixXf live_keypoints = core::extractHighDensityKeypoints(
        density_map, x_edges, y_edges, live_keypoint_density_threshold_
    );
    
    // Perform registration
    float x_min = x_edges(0);
    float x_max = x_edges(x_edges.size() - 1);
    float y_min = y_edges(0);
    float y_max = y_edges(y_edges.size() - 1);
    
    Eigen::VectorXf initial_x_edges = Eigen::VectorXf::LinSpaced(2, x_min, x_max);
    Eigen::VectorXf initial_y_edges = Eigen::VectorXf::LinSpaced(2, y_min, y_max);
    
    core::TransformResult result = core::hierarchicalAdaptiveSearch(
        global_map_keypoints_,
        live_keypoints,
        initial_x_edges,
        initial_y_edges,
        level_configs_,
        num_candidates_per_level_,
        min_candidate_separation_factor_,
        global_grid_size_,
        num_processes_
    );
    
    return std::make_tuple(live_keypoints, result);
}

void HiggsrServerNode::registerScanCallback(
    const std::shared_ptr<higgsr_interface::srv::RegisterScan::Request> request,
    std::shared_ptr<higgsr_interface::srv::RegisterScan::Response> response) {
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    if (!global_map_kdtree_) {
        response->success = false;
        response->message = "Global map not loaded";
        return;
    }
    
    try {
        std::lock_guard<std::mutex> lock(registration_mutex_);
        
        // Process the scan
        auto [live_keypoints, result] = processLiveScan(request->live_scan_info.point_cloud);
        
        if (result.valid) {
            // Create transform matrix
            Eigen::Matrix4f transform = core::createTransformMatrix4x4(
                result.tx, result.ty, result.theta_deg
            );
            
            // Convert to ROS message
            geometry_msgs::msg::Transform tf_msg = tf2::eigenToTransform(
                Eigen::Isometry3d(transform.cast<double>())
            ).transform;
            
            // Create TransformStamped message
            geometry_msgs::msg::TransformStamped tf_stamped;
            tf_stamped.header.stamp = this->now();
            tf_stamped.header.frame_id = global_frame_id_;
            tf_stamped.child_frame_id = "base_link";
            tf_stamped.transform = tf_msg;
            
            response->estimated_transform = tf_stamped;
            response->score = result.score;
            response->success = true;
            response->message = "Registration successful";
            
            // Publish visualization
            publishVisualization(live_keypoints, result);
            
        } else {
            response->success = false;
            response->message = "Registration failed - no valid match found";
        }
        
    } catch (const std::exception& e) {
        response->success = false;
        response->message = std::string("Exception: ") + e.what();
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
        end_time - start_time);
    
    // Store processing time in message
    auto processing_time_ms = duration.count();
    
    RCLCPP_INFO(this->get_logger(), 
                "Registration completed in %ld ms with score: %.1f",
                processing_time_ms, response->score);
}

void HiggsrServerNode::setGlobalMapCallback(
    const std::shared_ptr<higgsr_interface::srv::SetGlobalMap::Request> request,
    std::shared_ptr<higgsr_interface::srv::SetGlobalMap::Response> response) {
    
    // This is a legacy service - we load the map from file on startup
    response->success = false;
    response->message = "SetGlobalMap service is deprecated. Set global_map_file_path parameter instead.";
}

void HiggsrServerNode::publishVisualization(
    const Eigen::MatrixXf& live_scan_keypoints,
    const core::TransformResult& result) {
    
    // Publish transform
    geometry_msgs::msg::TransformStamped tf_stamped;
    tf_stamped.header.stamp = this->now();
    tf_stamped.header.frame_id = global_frame_id_;
    tf_stamped.child_frame_id = "base_link";
    tf_stamped.transform = tf2::eigenToTransform(
        Eigen::Isometry3d(core::createTransformMatrix4x4(
            result.tx, result.ty, result.theta_deg
        ).cast<double>())
    ).transform;
    
    transform_pub_->publish(tf_stamped);
    tf_broadcaster_->sendTransform(tf_stamped);
    
    // TODO: Publish keypoint markers
}

} // namespace nodes
} // namespace higgs_rc

// Component registration is optional
// #include "rclcpp_components/register_node_macro.hpp"
// RCLCPP_COMPONENTS_REGISTER_NODE(higgs_rc::nodes::HiggsrServerNode)
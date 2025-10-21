#include "higgs_rc/core/feature_extraction.hpp"
#include <cmath>
#include <vector>
#include <omp.h>

namespace higgs_rc {
namespace core {

Eigen::MatrixXf extractHighDensityKeypoints(
    const Eigen::MatrixXf& density_map,
    const Eigen::VectorXf& x_edges,
    const Eigen::VectorXf& y_edges,
    float density_threshold) {
    
    if (density_map.size() == 0) {
        return Eigen::MatrixXf(0, 2);
    }
    
    // Compute cell centers
    Eigen::VectorXf cell_centers_x(x_edges.size() - 1);
    Eigen::VectorXf cell_centers_y(y_edges.size() - 1);
    
    for (int i = 0; i < x_edges.size() - 1; ++i) {
        cell_centers_x(i) = (x_edges(i) + x_edges(i + 1)) / 2.0f;
    }
    
    for (int j = 0; j < y_edges.size() - 1; ++j) {
        cell_centers_y(j) = (y_edges(j) + y_edges(j + 1)) / 2.0f;
    }
    
    // Extract keypoints
    std::vector<Eigen::Vector2f> keypoints;
    keypoints.reserve(density_map.size() / 4); // Reserve some reasonable capacity
    
    for (int i = 0; i < density_map.rows(); ++i) {
        for (int j = 0; j < density_map.cols(); ++j) {
            if (density_map(i, j) > density_threshold) {
                keypoints.emplace_back(cell_centers_x(i), cell_centers_y(j));
            }
        }
    }
    
    // Convert to matrix format
    Eigen::MatrixXf result(keypoints.size(), 2);
    for (size_t i = 0; i < keypoints.size(); ++i) {
        result.row(i) = keypoints[i];
    }
    
    return result;
}

Eigen::MatrixXf applyTransformToKeypoints(
    const Eigen::MatrixXf& keypoints,
    float tx,
    float ty,
    float theta_rad) {
    
    if (keypoints.rows() == 0) {
        return Eigen::MatrixXf(0, 2);
    }
    
    const float cos_t = std::cos(theta_rad);
    const float sin_t = std::sin(theta_rad);
    
    // Create rotation matrix
    Eigen::Matrix2f rotation;
    rotation << cos_t, -sin_t,
                sin_t,  cos_t;
    
    // Apply transformation
    Eigen::MatrixXf transformed = (rotation * keypoints.transpose()).transpose();
    transformed.col(0).array() += tx;
    transformed.col(1).array() += ty;
    
    return transformed;
}

void applyTransformToKeypointsSIMD(
    const Eigen::MatrixXf& keypoints,
    float tx,
    float ty,
    float theta_rad,
    Eigen::MatrixXf& transformed_keypoints) {
    
    if (keypoints.rows() == 0) {
        transformed_keypoints = Eigen::MatrixXf(0, 2);
        return;
    }
    
    const float cos_t = std::cos(theta_rad);
    const float sin_t = std::sin(theta_rad);
    
    // Ensure output is properly sized
    if (transformed_keypoints.rows() != keypoints.rows() || 
        transformed_keypoints.cols() != 2) {
        transformed_keypoints.resize(keypoints.rows(), 2);
    }
    
    // Use OpenMP for parallel processing
    #pragma omp parallel for
    for (int i = 0; i < keypoints.rows(); ++i) {
        const float x = keypoints(i, 0);
        const float y = keypoints(i, 1);
        transformed_keypoints(i, 0) = x * cos_t - y * sin_t + tx;
        transformed_keypoints(i, 1) = x * sin_t + y * cos_t + ty;
    }
}

} // namespace core
} // namespace higgs_rc
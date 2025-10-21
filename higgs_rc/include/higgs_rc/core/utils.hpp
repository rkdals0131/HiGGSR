#ifndef HIGGS_RC_CORE_UTILS_HPP
#define HIGGS_RC_CORE_UTILS_HPP

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <string>
#include <tuple>

namespace higgs_rc {
namespace core {

/**
 * @brief Load point cloud from file using PCL
 * 
 * @param filepath Path to point cloud file (PLY, PCD, etc.)
 * @return Eigen::MatrixXf Loaded point cloud (Nx3)
 */
Eigen::MatrixXf loadPointCloudFromFile(const std::string& filepath);

/**
 * @brief Create 2.5D height variance map from 3D point cloud
 * 
 * @param points_3d 3D point cloud (Nx3)
 * @param grid_cell_size Grid cell size in meters
 * @param min_points_per_cell Minimum points per cell for density calculation
 * @param density_metric Density metric ("std" or "range")
 * @return std::tuple<Eigen::MatrixXf, Eigen::VectorXf, Eigen::VectorXf> 
 *         Density map, x edges, y edges
 */
std::tuple<Eigen::MatrixXf, Eigen::VectorXf, Eigen::VectorXf> 
create2DHeightVarianceMap(
    const Eigen::MatrixXf& points_3d,
    float grid_cell_size,
    int min_points_per_cell = 1,
    const std::string& density_metric = "std");

/**
 * @brief Create 4x4 transformation matrix from 2D transformation parameters
 * 
 * @param tx Translation in x direction
 * @param ty Translation in y direction
 * @param theta_deg Rotation angle in degrees
 * @return Eigen::Matrix4f 4x4 homogeneous transformation matrix
 */
Eigen::Matrix4f createTransformMatrix4x4(float tx, float ty, float theta_deg);

/**
 * @brief Convert degrees to radians
 */
inline float deg2rad(float degrees) {
    return degrees * M_PI / 180.0f;
}

/**
 * @brief Convert radians to degrees
 */
inline float rad2deg(float radians) {
    return radians * 180.0f / M_PI;
}

/**
 * @brief Compute statistics for binned data
 */
struct BinnedStatistics {
    Eigen::MatrixXf count;
    Eigen::MatrixXf mean;
    Eigen::MatrixXf std;
    Eigen::MatrixXf min;
    Eigen::MatrixXf max;
};

/**
 * @brief Compute 2D binned statistics
 */
BinnedStatistics computeBinnedStatistics2D(
    const Eigen::VectorXf& x,
    const Eigen::VectorXf& y,
    const Eigen::VectorXf& values,
    const Eigen::VectorXf& x_edges,
    const Eigen::VectorXf& y_edges);

} // namespace core
} // namespace higgs_rc

#endif // HIGGS_RC_CORE_UTILS_HPP
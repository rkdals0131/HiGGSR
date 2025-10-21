#ifndef HIGGS_RC_CORE_FEATURE_EXTRACTION_HPP
#define HIGGS_RC_CORE_FEATURE_EXTRACTION_HPP

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <vector>

namespace higgs_rc {
namespace core {

/**
 * @brief Extract keypoints from a density map based on a threshold
 * 
 * @param density_map 2D density map
 * @param x_edges Grid boundaries in x direction
 * @param y_edges Grid boundaries in y direction
 * @param density_threshold Minimum density threshold for keypoint extraction
 * @return Eigen::MatrixXf Extracted keypoints (Nx2)
 */
Eigen::MatrixXf extractHighDensityKeypoints(
    const Eigen::MatrixXf& density_map,
    const Eigen::VectorXf& x_edges,
    const Eigen::VectorXf& y_edges,
    float density_threshold);

/**
 * @brief Apply 2D transformation (translation and rotation) to keypoints
 * 
 * @param keypoints Keypoints to transform (Nx2)
 * @param tx Translation in x direction
 * @param ty Translation in y direction
 * @param theta_rad Rotation angle in radians
 * @return Eigen::MatrixXf Transformed keypoints (Nx2)
 */
Eigen::MatrixXf applyTransformToKeypoints(
    const Eigen::MatrixXf& keypoints,
    float tx,
    float ty,
    float theta_rad);

/**
 * @brief Batch apply transformation to keypoints with SIMD optimization
 * 
 * @param keypoints Keypoints to transform (Nx2)
 * @param tx Translation in x direction
 * @param ty Translation in y direction
 * @param theta_rad Rotation angle in radians
 * @param[out] transformed_keypoints Pre-allocated output matrix (Nx2)
 */
void applyTransformToKeypointsSIMD(
    const Eigen::MatrixXf& keypoints,
    float tx,
    float ty,
    float theta_rad,
    Eigen::MatrixXf& transformed_keypoints);

} // namespace core
} // namespace higgs_rc

#endif // HIGGS_RC_CORE_FEATURE_EXTRACTION_HPP
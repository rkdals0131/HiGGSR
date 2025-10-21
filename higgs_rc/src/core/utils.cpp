#include "higgs_rc/core/utils.hpp"
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/point_types.h>
#include <algorithm>
#include <unordered_map>
#include <cmath>

namespace higgs_rc {
namespace core {

Eigen::MatrixXf loadPointCloudFromFile(const std::string& filepath) {
    pcl::PointCloud<pcl::PointXYZ> cloud;
    
    // Determine file type and load
    if (filepath.find(".ply") != std::string::npos) {
        if (pcl::io::loadPLYFile<pcl::PointXYZ>(filepath, cloud) == -1) {
            return Eigen::MatrixXf(0, 3);
        }
    } else if (filepath.find(".pcd") != std::string::npos) {
        if (pcl::io::loadPCDFile<pcl::PointXYZ>(filepath, cloud) == -1) {
            return Eigen::MatrixXf(0, 3);
        }
    } else {
        return Eigen::MatrixXf(0, 3);
    }
    
    // Convert to Eigen matrix
    Eigen::MatrixXf points(cloud.size(), 3);
    for (size_t i = 0; i < cloud.size(); ++i) {
        points(i, 0) = cloud[i].x;
        points(i, 1) = cloud[i].y;
        points(i, 2) = cloud[i].z;
    }
    
    return points;
}

std::tuple<Eigen::MatrixXf, Eigen::VectorXf, Eigen::VectorXf> 
create2DHeightVarianceMap(
    const Eigen::MatrixXf& points_3d,
    float grid_cell_size,
    int min_points_per_cell,
    const std::string& density_metric) {
    
    if (points_3d.rows() == 0) {
        return std::make_tuple(
            Eigen::MatrixXf(0, 0),
            Eigen::VectorXf(0),
            Eigen::VectorXf(0)
        );
    }
    
    // Extract coordinates
    Eigen::VectorXf x_coords = points_3d.col(0);
    Eigen::VectorXf y_coords = points_3d.col(1);
    Eigen::VectorXf z_coords = points_3d.col(2);
    
    // Find bounds
    float x_min = x_coords.minCoeff();
    float x_max = x_coords.maxCoeff();
    float y_min = y_coords.minCoeff();
    float y_max = y_coords.maxCoeff();
    
    // Handle edge cases
    if (x_max == x_min) x_max = x_min + grid_cell_size;
    if (y_max == y_min) y_max = y_min + grid_cell_size;
    
    // Calculate grid dimensions
    int num_bins_x = std::max(1, static_cast<int>(std::ceil((x_max - x_min) / grid_cell_size)));
    int num_bins_y = std::max(1, static_cast<int>(std::ceil((y_max - y_min) / grid_cell_size)));
    
    // Create grid edges
    Eigen::VectorXf x_edges = Eigen::VectorXf::LinSpaced(num_bins_x + 1, x_min, x_min + num_bins_x * grid_cell_size);
    Eigen::VectorXf y_edges = Eigen::VectorXf::LinSpaced(num_bins_y + 1, y_min, y_min + num_bins_y * grid_cell_size);
    
    // Compute binned statistics
    BinnedStatistics stats = computeBinnedStatistics2D(x_coords, y_coords, z_coords, x_edges, y_edges);
    
    // Create density map based on metric
    Eigen::MatrixXf density_map(num_bins_x, num_bins_y);
    
    if (density_metric == "std") {
        for (int i = 0; i < num_bins_x; ++i) {
            for (int j = 0; j < num_bins_y; ++j) {
                if (stats.count(i, j) >= min_points_per_cell) {
                    density_map(i, j) = stats.std(i, j);
                } else {
                    density_map(i, j) = 0.0f;
                }
            }
        }
    } else if (density_metric == "range") {
        for (int i = 0; i < num_bins_x; ++i) {
            for (int j = 0; j < num_bins_y; ++j) {
                if (stats.count(i, j) >= min_points_per_cell) {
                    density_map(i, j) = stats.max(i, j) - stats.min(i, j);
                } else {
                    density_map(i, j) = 0.0f;
                }
            }
        }
    }
    
    return std::make_tuple(density_map, x_edges, y_edges);
}

Eigen::Matrix4f createTransformMatrix4x4(float tx, float ty, float theta_deg) {
    float theta_rad = deg2rad(theta_deg);
    float cos_t = std::cos(theta_rad);
    float sin_t = std::sin(theta_rad);
    
    Eigen::Matrix4f transform = Eigen::Matrix4f::Identity();
    transform(0, 0) = cos_t;
    transform(0, 1) = -sin_t;
    transform(0, 3) = tx;
    transform(1, 0) = sin_t;
    transform(1, 1) = cos_t;
    transform(1, 3) = ty;
    
    return transform;
}

BinnedStatistics computeBinnedStatistics2D(
    const Eigen::VectorXf& x,
    const Eigen::VectorXf& y,
    const Eigen::VectorXf& values,
    const Eigen::VectorXf& x_edges,
    const Eigen::VectorXf& y_edges) {
    
    int num_bins_x = x_edges.size() - 1;
    int num_bins_y = y_edges.size() - 1;
    
    BinnedStatistics stats;
    stats.count = Eigen::MatrixXf::Zero(num_bins_x, num_bins_y);
    stats.mean = Eigen::MatrixXf::Zero(num_bins_x, num_bins_y);
    stats.std = Eigen::MatrixXf::Zero(num_bins_x, num_bins_y);
    stats.min = Eigen::MatrixXf::Constant(num_bins_x, num_bins_y, std::numeric_limits<float>::max());
    stats.max = Eigen::MatrixXf::Constant(num_bins_x, num_bins_y, std::numeric_limits<float>::lowest());
    
    // First pass: compute count, sum, min, max
    Eigen::MatrixXf sum = Eigen::MatrixXf::Zero(num_bins_x, num_bins_y);
    
    for (int i = 0; i < x.size(); ++i) {
        // Find bin indices
        int bin_x = -1, bin_y = -1;
        
        for (int j = 0; j < num_bins_x; ++j) {
            if (x(i) >= x_edges(j) && x(i) < x_edges(j + 1)) {
                bin_x = j;
                break;
            }
        }
        
        for (int j = 0; j < num_bins_y; ++j) {
            if (y(i) >= y_edges(j) && y(i) < y_edges(j + 1)) {
                bin_y = j;
                break;
            }
        }
        
        // Handle edge case for maximum values
        if (bin_x == -1 && x(i) == x_edges(num_bins_x)) bin_x = num_bins_x - 1;
        if (bin_y == -1 && y(i) == y_edges(num_bins_y)) bin_y = num_bins_y - 1;
        
        if (bin_x >= 0 && bin_y >= 0) {
            stats.count(bin_x, bin_y) += 1;
            sum(bin_x, bin_y) += values(i);
            stats.min(bin_x, bin_y) = std::min(stats.min(bin_x, bin_y), values(i));
            stats.max(bin_x, bin_y) = std::max(stats.max(bin_x, bin_y), values(i));
        }
    }
    
    // Compute mean
    for (int i = 0; i < num_bins_x; ++i) {
        for (int j = 0; j < num_bins_y; ++j) {
            if (stats.count(i, j) > 0) {
                stats.mean(i, j) = sum(i, j) / stats.count(i, j);
            }
        }
    }
    
    // Second pass: compute standard deviation
    Eigen::MatrixXf sum_sq_diff = Eigen::MatrixXf::Zero(num_bins_x, num_bins_y);
    
    for (int i = 0; i < x.size(); ++i) {
        int bin_x = -1, bin_y = -1;
        
        for (int j = 0; j < num_bins_x; ++j) {
            if (x(i) >= x_edges(j) && x(i) < x_edges(j + 1)) {
                bin_x = j;
                break;
            }
        }
        
        for (int j = 0; j < num_bins_y; ++j) {
            if (y(i) >= y_edges(j) && y(i) < y_edges(j + 1)) {
                bin_y = j;
                break;
            }
        }
        
        if (bin_x == -1 && x(i) == x_edges(num_bins_x)) bin_x = num_bins_x - 1;
        if (bin_y == -1 && y(i) == y_edges(num_bins_y)) bin_y = num_bins_y - 1;
        
        if (bin_x >= 0 && bin_y >= 0 && stats.count(bin_x, bin_y) > 0) {
            float diff = values(i) - stats.mean(bin_x, bin_y);
            sum_sq_diff(bin_x, bin_y) += diff * diff;
        }
    }
    
    // Compute standard deviation
    for (int i = 0; i < num_bins_x; ++i) {
        for (int j = 0; j < num_bins_y; ++j) {
            if (stats.count(i, j) > 1) {
                stats.std(i, j) = std::sqrt(sum_sq_diff(i, j) / (stats.count(i, j) - 1));
            }
        }
    }
    
    return stats;
}

} // namespace core
} // namespace higgs_rc
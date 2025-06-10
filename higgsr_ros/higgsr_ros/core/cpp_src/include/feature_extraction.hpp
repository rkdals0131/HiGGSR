#ifndef HIGGSR_FEATURE_EXTRACTION_HPP
#define HIGGSR_FEATURE_EXTRACTION_HPP

#include <vector>
#include <Eigen/Dense>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

namespace higgsr_core {

/**
 * @brief 키포인트를 나타내는 구조체
 */
struct Keypoint {
    double x;
    double y;
    
    Keypoint() : x(0.0), y(0.0) {}
    Keypoint(double x_val, double y_val) : x(x_val), y(y_val) {}
};

/**
 * @brief Feature extraction 모듈의 입력 파라미터 구조체
 * 타입 안전성을 위해 모든 파라미터를 명시적으로 정의
 */
struct FeatureExtractionParams {
    double density_threshold;
    bool validate_input;
    
    FeatureExtractionParams() 
        : density_threshold(0.0), validate_input(true) {}
    
    // 파라미터 유효성 검증
    bool isValid() const {
        return density_threshold >= 0.0;
    }
};

/**
 * @brief 밀도 맵에서 고밀도 키포인트를 추출하는 함수
 * 
 * @param density_map 2D 밀도 맵 (row-major order)
 * @param rows 밀도 맵의 행 수
 * @param cols 밀도 맵의 열 수  
 * @param x_edges X 방향 그리드 경계값 배열
 * @param y_edges Y 방향 그리드 경계값 배열
 * @param x_edges_size X 경계값 배열 크기
 * @param y_edges_size Y 경계값 배열 크기
 * @param params Feature extraction 파라미터
 * @return 추출된 키포인트 벡터
 * 
 * @throws std::invalid_argument 입력 파라미터가 유효하지 않은 경우
 * @throws std::runtime_error 처리 중 오류가 발생한 경우
 */
std::vector<Keypoint> extractHighDensityKeypoints(
    const double* density_map,
    int rows, 
    int cols,
    const double* x_edges,
    const double* y_edges, 
    int x_edges_size,
    int y_edges_size,
    const FeatureExtractionParams& params
);

/**
 * @brief 키포인트에 2D 변환(이동 및 회전)을 적용하는 함수
 * 
 * @param keypoints 변환할 키포인트 벡터
 * @param tx X 방향 이동
 * @param ty Y 방향 이동  
 * @param theta_rad 회전 각도(라디안)
 * @return 변환된 키포인트 벡터
 * 
 * @throws std::invalid_argument 입력이 유효하지 않은 경우
 */
std::vector<Keypoint> applyTransformToKeypoints(
    const std::vector<Keypoint>& keypoints,
    double tx,
    double ty, 
    double theta_rad
);

/**
 * @brief Eigen 행렬을 사용한 키포인트 변환 (고성능 버전)
 * 
 * @param keypoints_matrix 키포인트 행렬 (Nx2)
 * @param transformation_matrix 변환 행렬 (3x3 homogeneous)
 * @return 변환된 키포인트 행렬
 */
Eigen::MatrixXd applyTransformToKeypointsEigen(
    const Eigen::MatrixXd& keypoints_matrix,
    const Eigen::Matrix3d& transformation_matrix
);

// TODO: PCL 포인트클라우드 처리 함수들 추가 예정
// - convertPCLToKeypoints()
// - filterKeypointsByHeight()
// - computeKeypointDensity()

} // namespace higgsr_core

#endif // HIGGSR_FEATURE_EXTRACTION_HPP 
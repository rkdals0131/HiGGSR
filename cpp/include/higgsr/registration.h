#pragma once

#include <vector>
#include <Eigen/Core>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <string>
#include <tuple> // For std::tuple

namespace higgsr {

// 최종 정합 결과를 담을 구조체 (앞으로 이동)
struct RegistrationResult {
    Eigen::Matrix4f transform;
    float score;
    double time_elapsed_sec;
    long long total_iterations;

    RegistrationResult() : transform(Eigen::Matrix4f::Identity()), score(-1.0f), time_elapsed_sec(0.0), total_iterations(0) {}
};

// 계층적 탐색의 각 레벨 설정을 위한 구조체 (앞으로 이동)
struct LevelConfig {
    std::vector<int> grid_division; // {num_x_cells, num_y_cells}
    std::string search_area_type;   // "full_map", "relative_to_map", "absolute_size"
    
    // search_area_type이 "relative_to_map"일 경우 사용 (0.0 ~ 1.0)
    // search_area_type이 "absolute_size"일 경우 사용 {width, height}
    std::vector<float> area_param; 

    // search_area_type이 "full_map" 또는 level 0일 경우 사용 {min_deg, max_deg}
    std::vector<float> theta_range_deg; 
    
    // search_area_type이 "relative_to_map" 또는 "absolute_size"이고 level > 0일 경우 사용
    // 이전 레벨의 theta 기준으로 상대적인 범위 {rel_min_deg, rel_max_deg}
    std::vector<float> theta_range_deg_relative; 
    
    int theta_search_steps;
    float correspondence_distance_threshold_factor;
    std::vector<int> tx_ty_search_steps_per_cell; // {tx_steps, ty_steps}

    LevelConfig() : grid_division({5,5}), search_area_type("full_map"), 
                    area_param({1.0f}), theta_range_deg({0.f, 359.f}),
                    theta_range_deg_relative({-10.f, 10.f}),
                    theta_search_steps(36), correspondence_distance_threshold_factor(2.0f),
                    tx_ty_search_steps_per_cell({5,5}) {}
};

/**
 * HiGGSR(Hierarchical Global Grid Search and Registration) C++ 구현
 * 3D 포인트 클라우드의 계층적 전역 정합을 위한 클래스
 */
class Registration {
public:
    Registration();
    ~Registration();

    /**
     * 계층적 적응형 전역 정합 수행 (새로운 시그니처)
     * 
     * @param global_map_keypoints 전역 맵 포인트 클라우드 (키포인트)
     * @param live_scan_keypoints 라이브 스캔 포인트 클라우드 (키포인트)
     * @param initial_map_x_edges 초기 탐색 영역 x 경계 [min_x, max_x]
     * @param initial_map_y_edges 초기 탐색 영역 y 경계 [min_y, max_y]
     * @param level_configs 각 계층 레벨에 대한 설정
     * @param num_candidates_to_select_per_level 각 레벨에서 다음 레벨로 전달할 후보 수
     * @param min_candidate_separation_factor 후보 간 최소 이격 계수 (셀 크기 기준)
     * @param base_grid_cell_size 기본 그리드 셀 크기 (실제 거리 단위)
     * @param num_processes 병렬 처리에 사용할 스레드 수 (0이면 순차 처리, -1이면 CPU 코어 수)
     * @return 최종 변환 행렬, 점수 및 기타 정보
     */
    RegistrationResult hierarchicalAdaptiveSearch(
        const pcl::PointCloud<pcl::PointXYZ>::Ptr& global_map_keypoints,
        const pcl::PointCloud<pcl::PointXYZ>::Ptr& live_scan_keypoints,
        const std::vector<float>& initial_map_x_edges, // {min_x, max_x}
        const std::vector<float>& initial_map_y_edges, // {min_y, max_y}
        const std::vector<LevelConfig>& level_configs,
        int num_candidates_to_select_per_level,
        float min_candidate_separation_factor,
        float base_grid_cell_size = 0.2f, // 기본값 유지
        int num_processes = 0 // 기본값 유지
    );

    /**
     * 2.5D 높이 분산 맵 생성
     * 
     * @param points_3d 3D 포인트 클라우드
     * @param grid_cell_size 그리드 셀 크기
     * @param min_points_per_cell 밀도 계산에 필요한 최소 포인트 수
     * @param density_metric 밀도 메트릭 ('std' 또는 'range')
     * @return 밀도 맵, x 및 y 경계
     */
    std::tuple<Eigen::MatrixXf, Eigen::VectorXf, Eigen::VectorXf> create2DHeightVarianceMap(
        const pcl::PointCloud<pcl::PointXYZ>::Ptr& points_3d,
        float grid_cell_size,
        int min_points_per_cell = 3,
        const std::string& density_metric = "std"
    );

    /**
     * 높은 밀도를 가진 키포인트 추출
     * 
     * @param density_map 밀도 맵
     * @param x_edges x 방향 경계
     * @param y_edges y 방향 경계
     * @param density_threshold 밀도 임계값
     * @return 키포인트 벡터
     */
    std::vector<Eigen::Vector2f> extractHighDensityKeypoints(
        const Eigen::MatrixXf& density_map,
        const Eigen::VectorXf& x_edges,
        const Eigen::VectorXf& y_edges,
        float density_threshold
    );

private:
    // 내부 설정 및 상태 변수들
    struct Config {
        int num_processes;
        float grid_cell_size;
        float correspondence_distance_threshold_factor;
        // 추가 구성 매개변수들...
    };
    
    Config config_;
};

} // namespace higgsr 
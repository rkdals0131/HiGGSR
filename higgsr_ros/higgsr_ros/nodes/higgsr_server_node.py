#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import numpy as np
import json
import os
from threading import Lock

from higgsr_interface.srv import RegisterScan, SetGlobalMap
from geometry_msgs.msg import TransformStamped
import tf_transformations

from higgsr_ros.core import utils as core_utils
from higgsr_ros.core import feature_extraction as core_feature_extraction
from higgsr_ros.core import registration as core_registration
from higgsr_ros.utils import ros_utils


class HiGGSRServerNode(Node):
    """
    HiGGSR 알고리즘을 실행하는 서버 노드
    - 시작 시 Data 디렉토리에서 글로벌 맵 자동 로드
    - 스캔 정합 서비스 제공
    """
    
    def __init__(self):
        super().__init__('higgsr_server_node')
        self.get_logger().info("HiGGSR 서버 노드 시작")

        # 내부 상태 변수
        self.global_map_points_3d = None
        self.density_map_global = None
        self.x_edges_global = None
        self.y_edges_global = None
        self.global_keypoints = None
        self.map_set = False
        self.processing_lock = Lock()

        # 파라미터 선언
        self._declare_node_parameters()

        # 변환 결과 퍼블리시를 위한 퍼블리셔 생성
        self.transform_publisher = self.create_publisher(
            TransformStamped,
            'higgsr_transform',
            10)

        # 글로벌 맵 자동 로드
        self._load_global_map_from_data_directory()

        # 서비스 서버 생성
        self.set_global_map_service = self.create_service(
            SetGlobalMap,
            'set_global_map',
            self.handle_set_global_map)
        
        self.register_scan_service = self.create_service(
            RegisterScan,
            'register_scan',
            self.handle_register_scan)

        self.get_logger().info("HiGGSR 서버 노드 초기화 완료")

    def _declare_node_parameters(self):
        """ROS 파라미터 선언"""
        # 글로벌 맵 파일 경로 파라미터 추가
        self.declare_parameter('global_map_file_path', 'src/HiGGSR/Data/around_singong - Cloud.ply')
        
        # 글로벌 맵 처리 파라미터
        self.declare_parameter('global_grid_size', 0.2)
        self.declare_parameter('global_min_points_for_density_calc', 3)
        self.declare_parameter('global_density_metric', 'std')
        self.declare_parameter('global_keypoint_density_threshold', 0.1)
        self.declare_parameter('global_frame_id', 'map_higgsr')

        # 라이브 스캔 처리 파라미터
        self.declare_parameter('live_grid_size', 0.2)
        self.declare_parameter('live_min_points_for_density_calc', 3)
        self.declare_parameter('live_density_metric', 'std')
        self.declare_parameter('live_keypoint_density_threshold', 0.1)

        # 등록 알고리즘 파라미터
        default_level_configs = [
            {
                "grid_division": [6, 6],
                "search_area_type": "full_map",
                "theta_range_deg": [0, 359],
                "theta_search_steps": 48,
                "correspondence_distance_threshold_factor": 7.0,
                "tx_ty_search_steps_per_cell": [10, 10]
            },
            {
                "grid_division": [7, 7],
                "search_area_type": "relative_to_map",
                "area_ratio_or_size": 0.4,
                "theta_range_deg_relative": [0, 359],
                "theta_search_steps": 48,
                "correspondence_distance_threshold_factor": 5.0,
                "tx_ty_search_steps_per_cell": [10, 10]
            },
            {
                "grid_division": [4, 4],
                "search_area_type": "absolute_size",
                "area_ratio_or_size": [40.0, 40.0],
                "theta_range_deg_relative": [0, 359],
                "theta_search_steps": 48,
                "correspondence_distance_threshold_factor": 2.5,
                "tx_ty_search_steps_per_cell": [10, 10]
            }
        ]
        
        self.declare_parameter('level_configs', json.dumps(default_level_configs))
        self.declare_parameter('num_candidates_per_level', 3)
        self.declare_parameter('min_candidate_separation_factor', 1.5)
        self.declare_parameter('num_processes', 0)
        
        # 시각화 파라미터 추가
        self.declare_parameter('enable_matplotlib_visualization', True)
        self.declare_parameter('enable_2d_keypoints_visualization', True)
        self.declare_parameter('enable_3d_result_visualization', True)
        self.declare_parameter('enable_super_grid_heatmap_visualization', False)

    def _load_global_map_from_data_directory(self):
        """Data 디렉토리에서 글로벌 맵을 자동으로 로드"""
        try:
            # 글로벌 맵 파일 경로 가져오기
            map_file_path = self.get_parameter('global_map_file_path').get_parameter_value().string_value
            
            # 절대 경로로 변환
            if not os.path.isabs(map_file_path):
                # 워크스페이스 루트에서 상대 경로로 계산
                workspace_root = '/home/user1/ROS2_Workspace/higgsros_ws'
                map_file_path = os.path.join(workspace_root, map_file_path)
            
            self.get_logger().info(f"글로벌 맵 로드 시도: {map_file_path}")
            
            if not os.path.exists(map_file_path):
                self.get_logger().error(f"글로벌 맵 파일을 찾을 수 없습니다: {map_file_path}")
                return
                
            # PLY 파일 로드
            self.global_map_points_3d = self._load_ply_file(map_file_path)
            
            if self.global_map_points_3d is None or self.global_map_points_3d.shape[0] == 0:
                self.get_logger().error("글로벌 맵 로드 실패 또는 빈 포인트 클라우드")
                return
                
            self.get_logger().info(f"글로벌 맵 로드 완료: {self.global_map_points_3d.shape[0]} 점")
            
            # 글로벌 맵 처리
            self._process_global_map()
            
        except Exception as e:
            self.get_logger().error(f"글로벌 맵 자동 로드 중 오류: {e}")
    
    def _load_ply_file(self, file_path):
        """PLY 파일을 로드하여 NumPy 배열로 반환"""
        try:
            import open3d as o3d
            pcd = o3d.io.read_point_cloud(file_path)
            if len(pcd.points) == 0:
                return None
            return np.asarray(pcd.points)
        except ImportError:
            self.get_logger().error("open3d가 설치되지 않았습니다. PLY 파일을 로드할 수 없습니다.")
            return None
        except Exception as e:
            self.get_logger().error(f"PLY 파일 로드 중 오류: {e}")
            return None
    
    def _process_global_map(self):
        """로드된 글로벌 맵을 처리하여 키포인트 추출"""
        try:
            # 파라미터 가져오기
            grid_size = self.get_parameter('global_grid_size').get_parameter_value().double_value
            min_points = self.get_parameter('global_min_points_for_density_calc').get_parameter_value().integer_value
            density_metric = self.get_parameter('global_density_metric').get_parameter_value().string_value
            keypoint_threshold = self.get_parameter('global_keypoint_density_threshold').get_parameter_value().double_value

            # 1. Pillar Map 생성
            self.density_map_global, self.x_edges_global, self.y_edges_global = \
                core_utils.create_2d_height_variance_map(
                    self.global_map_points_3d,
                    grid_cell_size=grid_size,
                    min_points_per_cell=min_points,
                    density_metric=density_metric)

            if self.density_map_global.size == 0:
                raise ValueError("Pillar Map 생성 실패")

            self.get_logger().info(f"글로벌 Pillar Map 생성 완료: {self.density_map_global.shape}")

            # 2. 키포인트 추출
            self.global_keypoints = core_feature_extraction.extract_high_density_keypoints(
                self.density_map_global,
                self.x_edges_global,
                self.y_edges_global,
                density_threshold=keypoint_threshold)

            self.get_logger().info(f"글로벌 키포인트 추출 완료: {self.global_keypoints.shape[0]} 개")

            if self.global_keypoints.shape[0] == 0:
                self.get_logger().warn("추출된 글로벌 키포인트가 없습니다")

            self.map_set = True
            self.get_logger().info("글로벌 맵 처리 및 설정 완료")

        except Exception as e:
            self.get_logger().error(f"글로벌 맵 처리 중 오류: {e}")
            self.map_set = False

    def handle_set_global_map(self, request, response):
        """글로벌 맵 설정 서비스 핸들러"""
        self.get_logger().info("글로벌 맵 설정 요청 수신")
        
        with self.processing_lock:
            try:
                # 포인트 클라우드 변환
                cloud_msg = request.global_map_info.point_cloud
                points_3d = ros_utils.convert_ros_point_cloud2_to_numpy(
                    cloud_msg, field_names=('x', 'y', 'z'))
                
                if points_3d is None or points_3d.shape[0] == 0:
                    raise ValueError("포인트 클라우드가 비어 있거나 변환 실패")

                self.global_map_points_3d = points_3d
                self.get_logger().info(f"글로벌 맵 포인트 수신: {points_3d.shape[0]} 점")

                # 파라미터 가져오기
                grid_size = self.get_parameter('global_grid_size').get_parameter_value().double_value
                min_points = self.get_parameter('global_min_points_for_density_calc').get_parameter_value().integer_value
                density_metric = self.get_parameter('global_density_metric').get_parameter_value().string_value
                keypoint_threshold = self.get_parameter('global_keypoint_density_threshold').get_parameter_value().double_value

                # 1. Pillar Map 생성
                self.density_map_global, self.x_edges_global, self.y_edges_global = \
                    core_utils.create_2d_height_variance_map(
                        self.global_map_points_3d,
                        grid_cell_size=grid_size,
                        min_points_per_cell=min_points,
                        density_metric=density_metric)

                if self.density_map_global.size == 0:
                    raise ValueError("Pillar Map 생성 실패")

                self.get_logger().info(f"글로벌 Pillar Map 생성 완료: {self.density_map_global.shape}")

                # 2. 키포인트 추출
                self.global_keypoints = core_feature_extraction.extract_high_density_keypoints(
                    self.density_map_global,
                    self.x_edges_global,
                    self.y_edges_global,
                    density_threshold=keypoint_threshold)

                self.get_logger().info(f"글로벌 키포인트 추출 완료: {self.global_keypoints.shape[0]} 개")

                if self.global_keypoints.shape[0] == 0:
                    self.get_logger().warn("추출된 글로벌 키포인트가 없습니다")

                self.map_set = True
                response.success = True
                response.message = "글로벌 맵 설정 성공"

            except Exception as e:
                self.get_logger().error(f"글로벌 맵 설정 중 오류: {e}")
                response.success = False
                response.message = f"글로벌 맵 설정 실패: {str(e)}"

        return response

    def handle_register_scan(self, request, response):
        """스캔 등록 서비스 핸들러"""
        self.get_logger().info("스캔 등록 요청 수신")

        if not self.map_set:
            response.success = False
            response.message = "글로벌 맵이 설정되지 않음"
            return response

        with self.processing_lock:
            try:
                # 라이브 스캔 처리
                cloud_msg = request.live_scan_info.point_cloud
                live_scan_points = ros_utils.convert_ros_point_cloud2_to_numpy(
                    cloud_msg, field_names=('x', 'y', 'z'))

                if live_scan_points is None or live_scan_points.shape[0] == 0:
                    raise ValueError("라이브 스캔이 비어 있거나 변환 실패")

                self.get_logger().info(f"라이브 스캔 수신: {live_scan_points.shape[0]} 점")

                # 라이브 스캔 처리 파라미터
                live_grid_size = self.get_parameter('live_grid_size').get_parameter_value().double_value
                live_min_points = self.get_parameter('live_min_points_for_density_calc').get_parameter_value().integer_value
                live_density_metric = self.get_parameter('live_density_metric').get_parameter_value().string_value
                live_keypoint_threshold = self.get_parameter('live_keypoint_density_threshold').get_parameter_value().double_value

                # 라이브 스캔 Pillar Map 생성
                density_map_scan, x_edges_scan, y_edges_scan = \
                    core_utils.create_2d_height_variance_map(
                        live_scan_points,
                        grid_cell_size=live_grid_size,
                        min_points_per_cell=live_min_points,
                        density_metric=live_density_metric)

                if density_map_scan.size == 0:
                    raise ValueError("라이브 스캔 Pillar Map 생성 실패")

                # 라이브 스캔 키포인트 추출
                scan_keypoints = core_feature_extraction.extract_high_density_keypoints(
                    density_map_scan, x_edges_scan, y_edges_scan,
                    density_threshold=live_keypoint_threshold)

                self.get_logger().info(f"라이브 스캔 키포인트: {scan_keypoints.shape[0]} 개")

                if scan_keypoints.shape[0] == 0:
                    self.get_logger().warn("라이브 스캔 키포인트가 없습니다")

                # 등록 알고리즘 파라미터
                level_configs_str = self.get_parameter('level_configs').get_parameter_value().string_value
                level_configs = self.parse_level_configs(level_configs_str)
                num_candidates = self.get_parameter('num_candidates_per_level').get_parameter_value().integer_value
                min_separation = self.get_parameter('min_candidate_separation_factor').get_parameter_value().double_value
                num_processes_param = self.get_parameter('num_processes').get_parameter_value().integer_value
                num_processes = num_processes_param if num_processes_param > 0 else None

                # 글로벌 맵 경계 설정
                initial_map_x_edges = [self.x_edges_global[0], self.x_edges_global[-1]]
                initial_map_y_edges = [self.y_edges_global[0], self.y_edges_global[-1]]

                # 계층적 등록 수행
                self.get_logger().info("계층적 등록 시작...")
                final_transform_dict, final_score, all_levels_visualization_data, total_hierarchical_time, total_calc_iterations = core_registration.hierarchical_adaptive_search(
                    self.global_keypoints, scan_keypoints,
                    initial_map_x_edges, initial_map_y_edges,
                    level_configs, num_candidates, min_separation,
                    self.get_parameter('global_grid_size').get_parameter_value().double_value,
                    num_processes=num_processes)

                if final_transform_dict is None:
                    raise ValueError("유효한 변환을 찾지 못함")

                self.get_logger().info(f"등록 완료! 점수: {final_score:.3f}")
                self.get_logger().info(
                    f"변환: tx={final_transform_dict['tx']:.3f}, "
                    f"ty={final_transform_dict['ty']:.3f}, "
                    f"theta={final_transform_dict['theta_deg']:.1f}°")
                self.get_logger().info(f"총 계산 시간: {total_hierarchical_time:.2f}초, 총 반복 횟수: {total_calc_iterations}")

                # matplotlib 시각화 (옵션)
                if self.get_parameter('enable_matplotlib_visualization').get_parameter_value().bool_value:
                    self._perform_matplotlib_visualization(
                        live_scan_points, scan_keypoints, final_transform_dict, 
                        final_score, all_levels_visualization_data, level_configs)

                # 변환 결과를 토픽으로 퍼블리시 (시각화 노드에서 사용)
                self._publish_transform_result(final_transform_dict, request.live_scan_info)
                
                # === Open3D vs RViz 변환 비교 로깅 ===
                self.get_logger().info("=== Open3D vs RViz 변환 비교 ===")
                self.get_logger().info("동일한 변환 파라미터가 두 시각화에 사용되었는지 확인:")
                self.get_logger().info(f"  변환 파라미터: tx={final_transform_dict['tx']:.6f}, ty={final_transform_dict['ty']:.6f}, theta_deg={final_transform_dict['theta_deg']:.6f}")
                
                # 샘플 포인트로 두 방식의 변환 결과 비교
                if live_scan_points is not None and live_scan_points.shape[0] > 0:
                    sample_points = live_scan_points[:3]  # 첫 3개 포인트만 비교
                    
                    # 1. Open3D/matplotlib 방식 (4x4 행렬)
                    transform_matrix_4x4 = core_utils.create_transform_matrix_4x4(
                        final_transform_dict['tx'], final_transform_dict['ty'], final_transform_dict['theta_deg'])
                    points_homogeneous = np.hstack([sample_points, np.ones((sample_points.shape[0], 1))])
                    open3d_transformed = (transform_matrix_4x4 @ points_homogeneous.T).T[:, :3]
                    
                    # 2. RViz 방식 (쿼터니언 변환, 시각화 노드에서 사용할 방식과 동일)
                    # 먼저 쿼터니언 생성
                    theta_rad = np.deg2rad(final_transform_dict['theta_deg'])
                    quat = tf_transformations.quaternion_from_euler(0, 0, theta_rad)
                    
                    # 쿼터니언을 다시 오일러로 변환 (시각화 노드에서 하는 과정과 동일)
                    recovered_euler = tf_transformations.euler_from_quaternion(quat)
                    recovered_theta_deg = np.rad2deg(recovered_euler[2])
                    
                    # 복원된 각도로 4x4 행렬 생성
                    rviz_transform_matrix_4x4 = core_utils.create_transform_matrix_4x4(
                        final_transform_dict['tx'], final_transform_dict['ty'], recovered_theta_deg)
                    rviz_transformed = (rviz_transform_matrix_4x4 @ points_homogeneous.T).T[:, :3]
                    
                    # 결과 비교
                    self.get_logger().info("변환 결과 비교 (샘플 포인트):")
                    for i in range(min(3, sample_points.shape[0])):
                        orig = sample_points[i]
                        open3d_result = open3d_transformed[i]
                        rviz_result = rviz_transformed[i]
                        diff = rviz_result - open3d_result
                        
                        self.get_logger().info(f"  점[{i}]: 원본({orig[0]:.3f}, {orig[1]:.3f}, {orig[2]:.3f})")
                        self.get_logger().info(f"    Open3D 변환 후: ({open3d_result[0]:.6f}, {open3d_result[1]:.6f}, {open3d_result[2]:.6f})")
                        self.get_logger().info(f"    RViz 변환 후:   ({rviz_result[0]:.6f}, {rviz_result[1]:.6f}, {rviz_result[2]:.6f})")
                        self.get_logger().info(f"    차이:          ({diff[0]:.6f}, {diff[1]:.6f}, {diff[2]:.6f})")
                        
                    # 각도 비교
                    self.get_logger().info(f"각도 비교:")
                    self.get_logger().info(f"  원본 theta_deg: {final_transform_dict['theta_deg']:.6f}°")
                    self.get_logger().info(f"  쿼터니언 경유 후 복원된 theta_deg: {recovered_theta_deg:.6f}°")
                    self.get_logger().info(f"  각도 차이: {recovered_theta_deg - final_transform_dict['theta_deg']:.6f}°")
                    
                self.get_logger().info("=== Open3D vs RViz 변환 비교 완료 ===")

                # 응답 생성
                transform_stamped = self.create_transform_stamped(
                    final_transform_dict, request.live_scan_info)

                response.success = True
                response.estimated_transform = transform_stamped
                response.score = float(final_score) if final_score is not None else 0.0
                response.message = "스캔 등록 성공"

            except Exception as e:
                self.get_logger().error(f"스캔 등록 중 오류: {e}")
                response.success = False
                response.message = f"스캔 등록 실패: {str(e)}"

        return response

    def parse_level_configs(self, level_configs_str):
        """레벨 설정 JSON 파싱"""
        try:
            level_configs = json.loads(level_configs_str)
            for config in level_configs:
                for key, value in config.items():
                    if isinstance(value, list) and key in [
                        'grid_division', 'theta_range_deg', 'theta_range_deg_relative',
                        'tx_ty_search_steps_per_cell']:
                        config[key] = tuple(value)
                    elif key == 'area_ratio_or_size' and isinstance(value, list):
                        config[key] = tuple(value)
            return level_configs
        except json.JSONDecodeError as e:
            self.get_logger().error(f"level_configs JSON 파싱 오류: {e}")
            return None

    def _perform_matplotlib_visualization(self, live_scan_points, scan_keypoints, 
                                        transform_dict, score, all_levels_data, level_configs):
        """matplotlib을 사용한 시각화 수행"""
        try:
            from higgsr_ros.visualization import visualization as viz
            from higgsr_ros.core import feature_extraction as core_feature_extraction
            
            # 변환 파라미터 추출
            tx = transform_dict['tx']
            ty = transform_dict['ty']
            theta_deg = transform_dict['theta_deg']
            
            self.get_logger().info("=== Open3D/matplotlib 시각화 변환 정보 ===")
            self.get_logger().info(f"사용된 변환 파라미터: tx={tx:.6f}, ty={ty:.6f}, theta_deg={theta_deg:.6f}")
            
            # 4x4 변환 행렬 생성 및 로깅
            transform_matrix_4x4 = core_utils.create_transform_matrix_4x4(tx, ty, theta_deg)
            self.get_logger().info(f"Open3D용 4x4 변환 행렬:\n{transform_matrix_4x4}")
            
            # 2D 키포인트 정합 결과 시각화
            if self.get_parameter('enable_2d_keypoints_visualization').get_parameter_value().bool_value:
                self.get_logger().info("2D 키포인트 정합 결과 시각화 중...")
                try:
                    # 변환된 스캔 키포인트 계산
                    theta_rad = np.deg2rad(theta_deg)
                    transformed_scan_keypoints = core_feature_extraction.apply_transform_to_keypoints_numba(
                        scan_keypoints, tx, ty, theta_rad)
                    
                    # 키포인트 변환 샘플 로깅
                    if scan_keypoints.shape[0] > 0:
                        sample_idx = min(3, scan_keypoints.shape[0])
                        self.get_logger().info(f"키포인트 변환 샘플 (첫 {sample_idx}개):")
                        for i in range(sample_idx):
                            orig = scan_keypoints[i]
                            trans = transformed_scan_keypoints[i]
                            self.get_logger().info(f"  원본[{i}]: ({orig[0]:.3f}, {orig[1]:.3f}) → 변환후: ({trans[0]:.3f}, {trans[1]:.3f})")
                    
                    viz.visualize_2d_keypoint_registration(
                        self.global_keypoints, scan_keypoints, transformed_scan_keypoints,
                        self.x_edges_global, self.y_edges_global, 
                        title=f"Live Scan Registration (Score: {score:.3f})")
                        
                except Exception as e:
                    self.get_logger().error(f"2D 키포인트 시각화 오류: {e}")
            
            # 계층적 탐색 히트맵 시각화
            if (self.get_parameter('enable_super_grid_heatmap_visualization').get_parameter_value().bool_value and 
                all_levels_data):
                self.get_logger().info("계층적 탐색 히트맵 시각화 중...")
                self.get_logger().info(f"all_levels_data 타입: {type(all_levels_data)}")
                self.get_logger().info(f"all_levels_data 길이: {len(all_levels_data) if hasattr(all_levels_data, '__len__') else 'N/A'}")
                
                # all_levels_data가 반복 가능한 객체(리스트 등)인지 확인
                if hasattr(all_levels_data, '__iter__') and not isinstance(all_levels_data, (str, int)):
                    try:
                        for idx, level_data in enumerate(all_levels_data):
                            self.get_logger().info(f"처리 중인 레벨 인덱스: {idx}")
                            self.get_logger().info(f"level_data 타입: {type(level_data)}")
                            
                            if isinstance(level_data, dict) and 'level' in level_data:
                                level_num = level_data['level']
                                self.get_logger().info(f"  Level {level_num} 히트맵 시각화...")
                                next_config_index = level_num
                                next_lvl_cfg = level_configs[next_config_index] if next_config_index < len(level_configs) else None
                                
                                viz.visualize_super_grid_scores(
                                    self.density_map_global,
                                    self.x_edges_global,
                                    self.y_edges_global,
                                    level_data['all_raw_cell_infos_this_level'],
                                    level_data['searched_areas_details'],
                                    self.x_edges_global,
                                    self.y_edges_global,
                                    level_data['selected_candidates_after_nms'],
                                    next_lvl_cfg,
                                    (self.x_edges_global, self.y_edges_global),
                                    title_suffix=f"Live Scan Level {level_num}")
                            else:
                                self.get_logger().warn(f"예상치 못한 level_data 구조: {level_data}")
                                
                    except Exception as e:
                        self.get_logger().error(f"히트맵 시각화 오류: {e}")
                        import traceback
                        self.get_logger().error(f"상세 오류 정보:\n{traceback.format_exc()}")
                else:
                    self.get_logger().warn(f"all_levels_data가 반복 가능한 객체가 아닙니다: {type(all_levels_data)}. 히트맵 시각화를 건너뜁니다.")
            
            # 3D 포인트클라우드 정합 결과 시각화
            if (self.get_parameter('enable_3d_result_visualization').get_parameter_value().bool_value and 
                live_scan_points is not None and self.global_map_points_3d is not None):
                self.get_logger().info("3D 포인트클라우드 정합 결과 시각화 중...")
                try:
                    # 4x4 변환 행렬 사용
                    self.get_logger().info("Open3D 시각화에서 사용할 변환 행렬:")
                    self.get_logger().info(f"{transform_matrix_4x4}")
                    
                    # 포인트 변환 샘플 로깅
                    if live_scan_points.shape[0] > 0:
                        sample_idx = min(3, live_scan_points.shape[0])
                        self.get_logger().info(f"3D 포인트 변환 샘플 (Open3D용, 첫 {sample_idx}개):")
                        
                        # 수동으로 변환 적용해서 결과 확인
                        points_homogeneous = np.hstack([live_scan_points[:sample_idx], np.ones((sample_idx, 1))])
                        transformed_points_homogeneous = (transform_matrix_4x4 @ points_homogeneous.T).T
                        transformed_points = transformed_points_homogeneous[:, :3]
                        
                        for i in range(sample_idx):
                            orig = live_scan_points[i]
                            trans = transformed_points[i]
                            self.get_logger().info(f"  원본[{i}]: ({orig[0]:.3f}, {orig[1]:.3f}, {orig[2]:.3f}) → 변환후: ({trans[0]:.3f}, {trans[1]:.3f}, {trans[2]:.3f})")
                    
                    viz.visualize_3d_registration_o3d(
                        self.global_map_points_3d, live_scan_points, transform_matrix_4x4)
                        
                except Exception as e:
                    self.get_logger().error(f"3D 시각화 오류: {e}")
                    
            self.get_logger().info("=== Open3D/matplotlib 시각화 변환 정보 완료 ===")
                    
        except ImportError as e:
            self.get_logger().warn(f"시각화 라이브러리를 가져올 수 없습니다: {e}")
        except Exception as e:
            self.get_logger().error(f"시각화 처리 중 오류: {e}")

    def create_transform_stamped(self, transform_dict, live_scan_info):
        """TransformStamped 메시지 생성"""
        transform_stamped = TransformStamped()
        transform_stamped.header.stamp = self.get_clock().now().to_msg()
        
        global_frame_id = self.get_parameter('global_frame_id').get_parameter_value().string_value
        transform_stamped.header.frame_id = global_frame_id
        transform_stamped.child_frame_id = live_scan_info.frame_id if live_scan_info.frame_id else "base_link"
        
        transform_stamped.transform.translation.x = float(transform_dict['tx'])
        transform_stamped.transform.translation.y = float(transform_dict['ty'])
        transform_stamped.transform.translation.z = 0.0

        # 쿼터니언 생성 과정 디버깅
        theta_deg = transform_dict['theta_deg']
        theta_rad = np.deg2rad(theta_deg)
        
        self.get_logger().info(f"=== 서버: 쿼터니언 생성 디버깅 ===")
        self.get_logger().info(f"원본 theta_deg: {theta_deg:.3f}°")
        self.get_logger().info(f"theta_rad: {theta_rad:.6f}")
        
        # METHOD 1: tf_transformations 사용 (기존 방식)
        q_tf = tf_transformations.quaternion_from_euler(0, 0, theta_rad)
        self.get_logger().info(f"tf_transformations 쿼터니언: x={q_tf[0]:.6f}, y={q_tf[1]:.6f}, z={q_tf[2]:.6f}, w={q_tf[3]:.6f}")
        
        # METHOD 2: 순수 z축 회전 쿼터니언 직접 계산 (의심되는 문제 해결책)
        # 순수 z축 회전: q = [0, 0, sin(θ/2), cos(θ/2)]
        half_angle = theta_rad / 2.0
        q_direct = [0.0, 0.0, np.sin(half_angle), np.cos(half_angle)]
        self.get_logger().info(f"직접 계산 쿼터니언: x={q_direct[0]:.6f}, y={q_direct[1]:.6f}, z={q_direct[2]:.6f}, w={q_direct[3]:.6f}")
        
        # METHOD 3: 두 방법 비교
        diff = [q_tf[i] - q_direct[i] for i in range(4)]
        max_diff = max(abs(d) for d in diff)
        self.get_logger().info(f"두 방법 차이: x={diff[0]:.6f}, y={diff[1]:.6f}, z={diff[2]:.6f}, w={diff[3]:.6f}")
        self.get_logger().info(f"최대 차이: {max_diff:.6f}")
        
        # 직접 계산한 쿼터니언 사용 (테스트용)
        if max_diff > 1e-6:
            self.get_logger().warn(f"tf_transformations와 직접 계산 결과가 다릅니다! 직접 계산 값을 사용합니다.")
            q = q_direct
        else:
            self.get_logger().info("tf_transformations와 직접 계산 결과가 일치합니다.")
            q = q_tf
        
        # 역변환 테스트
        test_euler = tf_transformations.euler_from_quaternion(q)
        test_yaw_deg = np.rad2deg(test_euler[2])
        self.get_logger().info(f"역변환 오일러 (도): roll={np.rad2deg(test_euler[0]):.3f}°, pitch={np.rad2deg(test_euler[1]):.3f}°, yaw={test_yaw_deg:.3f}°")
        self.get_logger().info(f"=== 서버: 쿼터니언 생성 디버깅 완료 ===")
        
        transform_stamped.transform.rotation.x = q[0]
        transform_stamped.transform.rotation.y = q[1]
        transform_stamped.transform.rotation.z = q[2]
        transform_stamped.transform.rotation.w = q[3]

        return transform_stamped

    def _publish_transform_result(self, transform_dict, live_scan_info):
        """변환 결과를 토픽으로 퍼블리시"""
        self.get_logger().info("=== RViz 변환 퍼블리시 정보 ===")
        self.get_logger().info(f"퍼블리시할 변환 파라미터: tx={transform_dict['tx']:.6f}, ty={transform_dict['ty']:.6f}, theta_deg={transform_dict['theta_deg']:.6f}")
        
        transform_stamped = self.create_transform_stamped(transform_dict, live_scan_info)
        
        # 생성된 TransformStamped 메시지 내용 로깅
        self.get_logger().info("생성된 TransformStamped 메시지:")
        self.get_logger().info(f"  header.frame_id: {transform_stamped.header.frame_id}")
        self.get_logger().info(f"  child_frame_id: {transform_stamped.child_frame_id}")
        self.get_logger().info(f"  translation: x={transform_stamped.transform.translation.x:.6f}, y={transform_stamped.transform.translation.y:.6f}, z={transform_stamped.transform.translation.z:.6f}")
        self.get_logger().info(f"  rotation (quaternion): x={transform_stamped.transform.rotation.x:.6f}, y={transform_stamped.transform.rotation.y:.6f}, z={transform_stamped.transform.rotation.z:.6f}, w={transform_stamped.transform.rotation.w:.6f}")
        
        # 쿼터니언을 다시 오일러로 변환해서 확인
        quat = [
            transform_stamped.transform.rotation.x,
            transform_stamped.transform.rotation.y,
            transform_stamped.transform.rotation.z,
            transform_stamped.transform.rotation.w
        ]
        euler = tf_transformations.euler_from_quaternion(quat)
        recovered_yaw_deg = np.rad2deg(euler[2])
        self.get_logger().info(f"  역변환 확인 - 오일러 각도: roll={np.rad2deg(euler[0]):.3f}°, pitch={np.rad2deg(euler[1]):.3f}°, yaw={recovered_yaw_deg:.3f}°")
        
        self.transform_publisher.publish(transform_stamped)
        
        # 4x4 변환 행렬 생성하여 딕셔너리에 추가
        final_transform_matrix_4x4 = core_utils.create_transform_matrix_4x4(
            transform_dict['tx'], transform_dict['ty'], transform_dict['theta_deg'])
        
        # RViz용 4x4 변환 행렬 로깅
        self.get_logger().info("RViz용 4x4 변환 행렬 (참조용):")
        self.get_logger().info(f"{final_transform_matrix_4x4}")
        
        # 확장된 변환 정보 로그
        self.get_logger().info(f"변환 결과 퍼블리시 완료: tx={transform_dict['tx']:.3f}, ty={transform_dict['ty']:.3f}, theta={transform_dict['theta_deg']:.1f}°")
        self.get_logger().info("=== RViz 변환 퍼블리시 정보 완료 ===")


def main(args=None):
    rclpy.init(args=args)
    
    node = HiGGSRServerNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('키보드 인터럽트 수신, 종료 중...')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main() 
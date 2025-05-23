#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import numpy as np
import json
import os
import time
import multiprocessing

from higgsr_interface.srv import ProcessFiles
from geometry_msgs.msg import TransformStamped, PoseStamped
from sensor_msgs.msg import PointCloud2
from visualization_msgs.msg import MarkerArray
import tf_transformations

from higgsr_ros.core import utils as core_utils
from higgsr_ros.core import feature_extraction as core_feature_extraction
from higgsr_ros.core import registration as core_registration
from higgsr_ros.visualization import visualization as viz


class FileProcessorNode(Node):
    def __init__(self):
        super().__init__('file_processor_node')
        self.get_logger().info("파일 처리 노드 시작.")
        
        # 알고리즘 파라미터 선언
        self._declare_algorithm_parameters()
        
        # 시각화 파라미터 선언
        self._declare_visualization_parameters()
        
        # RViz2 시각화를 위한 퍼블리셔들 추가
        self._setup_rviz_publishers()
        
        # ProcessFiles 서비스 서버 생성
        self.process_files_service = self.create_service(
            ProcessFiles,
            'process_files',
            self.handle_process_files_request
        )
        self.get_logger().info("ProcessFiles 서비스 준비 완료.")

    def _setup_rviz_publishers(self):
        """RViz2 시각화를 위한 퍼블리셔들을 설정합니다."""
        from higgsr_ros.visualization.visualization_node import HiGGSRVisualizationNode
        
        # 시각화 노드 인스턴스 생성 (메서드만 사용)
        self.viz_helper = HiGGSRVisualizationNode.__new__(HiGGSRVisualizationNode)
        
        # 퍼블리셔 직접 생성
        self.global_map_publisher = self.create_publisher(PointCloud2, 'higgsr_global_map', 10)
        self.live_scan_publisher = self.create_publisher(PointCloud2, 'higgsr_live_scan', 10)
        self.global_keypoints_publisher = self.create_publisher(MarkerArray, 'higgsr_global_keypoints', 10)
        self.scan_keypoints_publisher = self.create_publisher(MarkerArray, 'higgsr_scan_keypoints', 10)
        self.pose_publisher = self.create_publisher(PoseStamped, 'higgsr_pose', 10)
        
        self.get_logger().info("RViz2 시각화 퍼블리셔 설정 완료")

    def _declare_algorithm_parameters(self):
        """알고리즘 관련 파라미터들을 선언합니다."""
        
        # 그리드 및 밀도 계산 설정
        self.declare_parameter('grid_size', 0.2)
        self.declare_parameter('min_points_for_density_calc', 3)
        self.declare_parameter('density_metric', 'std')  # 'std' or 'range'
        self.declare_parameter('keypoint_density_threshold', 0.1)
        
        # 멀티프로세싱 설정
        self.declare_parameter('num_processes', 0)  # 0이면 자동 설정
        
        # 계층적 탐색 설정
        default_level_configs_str = '''
        [
            { 
                "grid_division": [6, 6], 
                "search_area_type": "full_map", 
                "theta_range_deg": [0, 359], "theta_search_steps": 48, 
                "correspondence_distance_threshold_factor": 7.0,
                "tx_ty_search_steps_per_cell": [10, 10] 
            },
            { 
                "grid_division": [7, 7], 
                "search_area_type": "relative_to_map", "area_ratio_or_size": 0.4,
                "theta_range_deg_relative": [0, 359], "theta_search_steps": 48,
                "correspondence_distance_threshold_factor": 5.0,
                "tx_ty_search_steps_per_cell": [10, 10]
            },
            { 
                "grid_division": [4, 4], 
                "search_area_type": "absolute_size", "area_ratio_or_size": [40.0, 40.0],
                "theta_range_deg_relative": [0, 359], "theta_search_steps": 48,
                "correspondence_distance_threshold_factor": 2.5,
                "tx_ty_search_steps_per_cell": [10, 10]
            }
        ]
        '''
        self.declare_parameter('level_configs', default_level_configs_str)
        self.declare_parameter('num_candidates_per_level', 3)
        self.declare_parameter('min_candidate_separation_factor', 1.5)
        
        # 결과 프레임 설정
        self.declare_parameter('global_frame_id', 'map')
        self.declare_parameter('scan_frame_id', 'base_link')

    def _declare_visualization_parameters(self):
        """시각화 관련 파라미터들을 선언합니다."""
        self.declare_parameter('enable_pillar_maps_visualization', False)
        self.declare_parameter('enable_2d_keypoints_visualization', True)
        self.declare_parameter('enable_super_grid_heatmap_visualization', True)
        self.declare_parameter('enable_3d_result_visualization', True)

    def _parse_level_configs(self, level_configs_str):
        """레벨 설정 JSON 문자열을 파싱합니다."""
        try:
            level_configs = json.loads(level_configs_str)
            # JSON 파싱 후 튜플 변환
            for config in level_configs:
                for key, value in config.items():
                    if isinstance(value, list) and key in [
                        'grid_division', 'theta_range_deg', 'theta_range_deg_relative', 
                        'tx_ty_search_steps_per_cell'
                    ]:
                        config[key] = tuple(value)
                    elif key == 'area_ratio_or_size' and isinstance(value, list):
                        config[key] = tuple(value)
            return level_configs
        except json.JSONDecodeError as e:
            self.get_logger().error(f"level_configs JSON 파싱 오류: {e}")
            return None

    def handle_process_files_request(self, request, response):
        """ProcessFiles 서비스 요청을 처리합니다."""
        self.get_logger().info(f"파일 처리 요청 수신:")
        self.get_logger().info(f"  전역 맵: {request.global_map_filepath}")
        self.get_logger().info(f"  라이브 스캔: {request.live_scan_filepath}")
        
        start_time = time.time()
        
        try:
            # 1. 파라미터 로드 (요청에 JSON이 있으면 사용, 없으면 노드 파라미터 사용)
            if request.algorithm_config_json.strip():
                self.get_logger().info("요청에서 제공된 알고리즘 설정 사용")
                config = self._load_config_from_request(request.algorithm_config_json)
                if config is None:
                    response.success = False
                    response.message = "알고리즘 설정 JSON 파싱 오류"
                    return response
            else:
                self.get_logger().info("노드 파라미터 사용")
                config = self._load_config_from_parameters()
            
            # 2. 파일 로딩
            self.get_logger().info("데이터 로딩 중...")
            if not os.path.exists(request.global_map_filepath):
                response.success = False
                response.message = f"전역 맵 파일을 찾을 수 없음: {request.global_map_filepath}"
                return response
            
            if not os.path.exists(request.live_scan_filepath):
                response.success = False
                response.message = f"라이브 스캔 파일을 찾을 수 없음: {request.live_scan_filepath}"
                return response
            
            global_map_points_3d = core_utils.load_point_cloud_from_file(request.global_map_filepath)
            live_scan_points_3d = core_utils.load_point_cloud_from_file(request.live_scan_filepath)
            
            if global_map_points_3d.shape[0] == 0 or live_scan_points_3d.shape[0] == 0:
                response.success = False
                response.message = "포인트 클라우드 로딩 실패"
                return response
                
            self.get_logger().info(f"데이터 로딩 완료: 전역맵 {global_map_points_3d.shape[0]}점, 스캔 {live_scan_points_3d.shape[0]}점")
            
            # 3. Pillar Map 생성
            self.get_logger().info("Pillar Map 생성 중...")
            density_map_global, x_edges_global, y_edges_global = core_utils.create_2d_height_variance_map(
                global_map_points_3d, config['grid_size'], 
                config['min_points_for_density_calc'], config['density_metric']
            )
            density_map_scan, x_edges_scan, y_edges_scan = core_utils.create_2d_height_variance_map(
                live_scan_points_3d, config['grid_size'], 
                config['min_points_for_density_calc'], config['density_metric']
            )
            
            if density_map_global.size == 0 or density_map_scan.size == 0:
                response.success = False
                response.message = "Pillar Map 생성 실패"
                return response
                
            self.get_logger().info(f"Pillar Map 생성 완료: 전역맵 {density_map_global.shape}, 스캔 {density_map_scan.shape}")
            
            # Pillar Map 시각화 (옵션)
            if config['visualization']['enable_pillar_maps']:
                viz.visualize_density_map(density_map_global, x_edges_global, y_edges_global, 
                                        title_suffix=f"Global Map ({config['density_metric']})")
                viz.visualize_density_map(density_map_scan, x_edges_scan, y_edges_scan, 
                                        title_suffix=f"Live Scan ({config['density_metric']})")
            
            # 4. 키포인트 추출
            self.get_logger().info("키포인트 추출 중...")
            global_keypoints = core_feature_extraction.extract_high_density_keypoints(
                density_map_global, x_edges_global, y_edges_global, config['keypoint_density_threshold']
            )
            scan_keypoints = core_feature_extraction.extract_high_density_keypoints(
                density_map_scan, x_edges_scan, y_edges_scan, config['keypoint_density_threshold']
            )
            
            self.get_logger().info(f"키포인트 추출 완료: 전역맵 {global_keypoints.shape[0]}개, 스캔 {scan_keypoints.shape[0]}개")
            
            if global_keypoints.shape[0] == 0 or scan_keypoints.shape[0] == 0:
                response.success = False
                response.message = "키포인트가 없어 정합 불가"
                return response
            
            # 5. 계층적 적응형 전역 정합 수행
            self.get_logger().info("계층적 적응형 전역 정합 수행 중...")
            
            initial_map_x_edges_for_search = [x_edges_global[0], x_edges_global[-1]]
            initial_map_y_edges_for_search = [y_edges_global[0], y_edges_global[-1]]
            
            final_transform_dict, final_score, all_levels_visualization_data, total_hierarchical_time, total_calc_iterations = core_registration.hierarchical_adaptive_search(
                global_keypoints, scan_keypoints,
                initial_map_x_edges_for_search, initial_map_y_edges_for_search,
                config['level_configs'],
                config['num_candidates_per_level'],
                config['min_candidate_separation_factor'],
                config['grid_size'],
                num_processes=config['num_processes']
            )
            
            if final_transform_dict is None:
                response.success = False
                response.message = "정합 실패: 유효한 변환을 찾지 못함"
                return response
            
            # 6. 결과 설정
            est_tx = final_transform_dict['tx']
            est_ty = final_transform_dict['ty']
            est_theta_deg = final_transform_dict['theta_deg']
            
            total_time = time.time() - start_time
            
            self.get_logger().info("--- 정합 결과 ---")
            self.get_logger().info(f"  추정된 변환: tx={est_tx:.3f}, ty={est_ty:.3f}, theta={est_theta_deg:.2f} deg")
            self.get_logger().info(f"  최고 점수: {final_score}")
            self.get_logger().info(f"  정합 소요 시간: {total_hierarchical_time:.2f} 초")
            self.get_logger().info(f"  총 계산된 변환 후보 수: {total_calc_iterations}")
            self.get_logger().info(f"  전체 소요 시간: {total_time:.2f} 초")
            
            # 변환 행렬 생성
            final_transform_matrix_4x4 = core_utils.create_transform_matrix_4x4(est_tx, est_ty, est_theta_deg)
            self.get_logger().info("최종 정합 결과 (4x4 동차 변환 행렬):")
            self.get_logger().info(f"\n{final_transform_matrix_4x4}")
            
            # 7. 시각화 (옵션)
            self._perform_visualizations(config, global_keypoints, scan_keypoints, 
                                       est_tx, est_ty, est_theta_deg, final_score,
                                       x_edges_global, y_edges_global,
                                       all_levels_visualization_data, density_map_global,
                                       global_map_points_3d, live_scan_points_3d,
                                       final_transform_matrix_4x4)
            
            # 8. RViz2 시각화 퍼블리시
            self._publish_to_rviz2(global_map_points_3d, live_scan_points_3d, 
                                 global_keypoints, scan_keypoints,
                                 {'tx': est_tx, 'ty': est_ty, 'theta_deg': est_theta_deg, 
                                  'transform_matrix': final_transform_matrix_4x4})
            
            # 9. 응답 설정
            response.success = True
            response.score = float(final_score)
            response.message = "정합 성공"
            response.processing_time_seconds = float(total_time)
            response.total_calc_iterations = int(total_calc_iterations)
            
            # TransformStamped 설정
            response.estimated_transform.header.stamp = self.get_clock().now().to_msg()
            response.estimated_transform.header.frame_id = config['global_frame_id']
            response.estimated_transform.child_frame_id = config['scan_frame_id']
            
            response.estimated_transform.transform.translation.x = float(est_tx)
            response.estimated_transform.transform.translation.y = float(est_ty)
            response.estimated_transform.transform.translation.z = 0.0
            
            q = tf_transformations.quaternion_from_euler(0, 0, np.deg2rad(est_theta_deg))
            response.estimated_transform.transform.rotation.x = q[0]
            response.estimated_transform.transform.rotation.y = q[1]
            response.estimated_transform.transform.rotation.z = q[2]
            response.estimated_transform.transform.rotation.w = q[3]
            
            return response
            
        except Exception as e:
            self.get_logger().error(f"파일 처리 중 예외 발생: {e}", exc_info=True)
            response.success = False
            response.message = f"처리 중 예외: {str(e)}"
            response.processing_time_seconds = float(time.time() - start_time)
            response.total_calc_iterations = 0
            return response

    def _load_config_from_request(self, config_json):
        """요청의 JSON에서 설정을 로드합니다."""
        try:
            config = json.loads(config_json)
            # 기본값 설정 및 누락된 값 보완
            defaults = self._get_default_config()
            for key, default_value in defaults.items():
                if key not in config:
                    config[key] = default_value
            
            # level_configs 파싱
            if 'level_configs' in config and isinstance(config['level_configs'], str):
                config['level_configs'] = self._parse_level_configs(config['level_configs'])
                if config['level_configs'] is None:
                    return None
            
            return config
        except json.JSONDecodeError as e:
            self.get_logger().error(f"요청 JSON 파싱 오류: {e}")
            return None

    def _load_config_from_parameters(self):
        """노드 파라미터에서 설정을 로드합니다."""
        config = {}
        
        # 알고리즘 파라미터
        config['grid_size'] = self.get_parameter('grid_size').get_parameter_value().double_value
        config['min_points_for_density_calc'] = self.get_parameter('min_points_for_density_calc').get_parameter_value().integer_value
        config['density_metric'] = self.get_parameter('density_metric').get_parameter_value().string_value
        config['keypoint_density_threshold'] = self.get_parameter('keypoint_density_threshold').get_parameter_value().double_value
        
        num_processes_param = self.get_parameter('num_processes').get_parameter_value().integer_value
        config['num_processes'] = num_processes_param if num_processes_param > 0 else None
        
        level_configs_str = self.get_parameter('level_configs').get_parameter_value().string_value
        config['level_configs'] = self._parse_level_configs(level_configs_str)
        
        config['num_candidates_per_level'] = self.get_parameter('num_candidates_per_level').get_parameter_value().integer_value
        config['min_candidate_separation_factor'] = self.get_parameter('min_candidate_separation_factor').get_parameter_value().double_value
        
        # 프레임 설정
        config['global_frame_id'] = self.get_parameter('global_frame_id').get_parameter_value().string_value
        config['scan_frame_id'] = self.get_parameter('scan_frame_id').get_parameter_value().string_value
        
        # 시각화 설정
        config['visualization'] = {
            'enable_pillar_maps': self.get_parameter('enable_pillar_maps_visualization').get_parameter_value().bool_value,
            'enable_2d_keypoints': self.get_parameter('enable_2d_keypoints_visualization').get_parameter_value().bool_value,
            'enable_super_grid_heatmap': self.get_parameter('enable_super_grid_heatmap_visualization').get_parameter_value().bool_value,
            'enable_3d_result': self.get_parameter('enable_3d_result_visualization').get_parameter_value().bool_value
        }
        
        return config

    def _get_default_config(self):
        """기본 설정 값들을 반환합니다."""
        return {
            'grid_size': 0.2,
            'min_points_for_density_calc': 3,
            'density_metric': 'std',
            'keypoint_density_threshold': 0.1,
            'num_processes': None,
            'num_candidates_per_level': 3,
            'min_candidate_separation_factor': 1.5,
            'global_frame_id': 'map',
            'scan_frame_id': 'base_link',
            'visualization': {
                'enable_pillar_maps': False,
                'enable_2d_keypoints': True,
                'enable_super_grid_heatmap': True,
                'enable_3d_result': True
            }
        }

    def _perform_visualizations(self, config, global_keypoints, scan_keypoints,
                              est_tx, est_ty, est_theta_deg, final_score,
                              x_edges_global, y_edges_global,
                              all_levels_visualization_data, density_map_global,
                              global_map_points_3d, live_scan_points_3d,
                              final_transform_matrix_4x4):
        """설정에 따라 시각화를 수행합니다."""
        
        if final_score <= -1:
            self.get_logger().warn("정합 점수가 낮아 시각화를 건너뜁니다.")
            return
        
        viz_config = config['visualization']
        
        # 2D 키포인트 정합 결과 시각화
        if viz_config['enable_2d_keypoints']:
            self.get_logger().info("2D 키포인트 정합 결과 시각화 중...")
            try:
                transformed_scan_keypoints = core_feature_extraction.apply_transform_to_keypoints_numba(
                    scan_keypoints, est_tx, est_ty, np.deg2rad(est_theta_deg)
                )
                viz.visualize_2d_keypoint_registration(
                    global_keypoints, scan_keypoints, transformed_scan_keypoints,
                    x_edges_global, y_edges_global, 
                    title=f"2D Keypoint Registration (Score: {final_score})"
                )
            except Exception as e:
                self.get_logger().error(f"2D 키포인트 시각화 오류: {e}")
        
        # 계층적 탐색 결과 히트맵 시각화
        if viz_config['enable_super_grid_heatmap'] and all_levels_visualization_data:
            self.get_logger().info("계층적 탐색 결과 히트맵 시각화 중...")
            try:
                for level_data in all_levels_visualization_data:
                    self.get_logger().info(f"  Level {level_data['level']} 히트맵 시각화...")
                    next_config_index = level_data['level']
                    next_lvl_cfg = config['level_configs'][next_config_index] if next_config_index < len(config['level_configs']) else None
                    
                    viz.visualize_super_grid_scores(
                        density_map_global,
                        x_edges_global,
                        y_edges_global,
                        level_data['all_raw_cell_infos_this_level'],
                        level_data['searched_areas_details'],
                        x_edges_global,
                        y_edges_global,
                        level_data['selected_candidates_after_nms'],
                        next_lvl_cfg,
                        (x_edges_global, y_edges_global),
                        title_suffix=f"Level {level_data['level']}"
                    )
            except Exception as e:
                self.get_logger().error(f"히트맵 시각화 오류: {e}")
        
        # 3D 포인트 클라우드 정합 결과 시각화
        if viz_config['enable_3d_result']:
            self.get_logger().info("3D 포인트 클라우드 정합 결과 시각화 중...")
            try:
                viz.visualize_3d_registration_o3d(
                    global_map_points_3d, live_scan_points_3d, final_transform_matrix_4x4
                )
            except Exception as e:
                self.get_logger().error(f"3D 시각화 오류: {e}")

    def _publish_to_rviz2(self, global_points_3d, live_scan_points_3d, global_keypoints_2d, scan_keypoints_2d, transform_result):
        """처리 결과를 RViz2로 퍼블리시합니다."""
        try:
            self.get_logger().info("RViz2 시각화 데이터 퍼블리시 시작...")
            
            # 시각화 헬퍼 메서드들을 직접 구현하여 사용
            frame_id = 'map'
            
            # 1. 글로벌 맵 포인트클라우드 퍼블리시
            if global_points_3d is not None and global_points_3d.shape[0] > 0:
                self._publish_point_cloud(global_points_3d, self.global_map_publisher, frame_id, (255, 255, 255))
                self.get_logger().info(f"글로벌 맵 포인트클라우드 퍼블리시: {global_points_3d.shape[0]} 포인트")
            
            # 2. 라이브 스캔 포인트클라우드 퍼블리시 (변환 적용)
            if live_scan_points_3d is not None and live_scan_points_3d.shape[0] > 0:
                transformed_scan = self._apply_transform_to_points(live_scan_points_3d, transform_result)
                self._publish_point_cloud(transformed_scan, self.live_scan_publisher, frame_id, (255, 0, 0))
                self.get_logger().info(f"변환된 스캔 포인트클라우드 퍼블리시: {transformed_scan.shape[0]} 포인트")
            
            # 3. 글로벌 키포인트 퍼블리시
            if global_keypoints_2d is not None and global_keypoints_2d.shape[0] > 0:
                self._publish_keypoints_as_markers(global_keypoints_2d, self.global_keypoints_publisher, 
                                                 "global_keypoints", frame_id, (0.0, 0.0, 1.0, 1.0), 0.8)
                self.get_logger().info(f"글로벌 키포인트 퍼블리시: {global_keypoints_2d.shape[0]} 개")
            
            # 4. 스캔 키포인트 퍼블리시 (변환 적용)
            if scan_keypoints_2d is not None and scan_keypoints_2d.shape[0] > 0:
                transformed_keypoints = self._apply_transform_to_keypoints(scan_keypoints_2d, transform_result)
                self._publish_keypoints_as_markers(transformed_keypoints, self.scan_keypoints_publisher, 
                                                 "scan_keypoints", frame_id, (0.0, 1.0, 0.0, 1.0), 0.6)
                self.get_logger().info(f"변환된 스캔 키포인트 퍼블리시: {transformed_keypoints.shape[0]} 개")
            
            # 5. 현재 포즈 퍼블리시
            pose_msg = PoseStamped()
            pose_msg.header.stamp = self.get_clock().now().to_msg()
            pose_msg.header.frame_id = frame_id
            pose_msg.pose.position.x = float(transform_result['tx'])
            pose_msg.pose.position.y = float(transform_result['ty'])
            pose_msg.pose.position.z = 0.0
            
            q = tf_transformations.quaternion_from_euler(0, 0, np.deg2rad(transform_result['theta_deg']))
            pose_msg.pose.orientation.x = q[0]
            pose_msg.pose.orientation.y = q[1]
            pose_msg.pose.orientation.z = q[2]
            pose_msg.pose.orientation.w = q[3]
            
            self.pose_publisher.publish(pose_msg)
            self.get_logger().info("현재 포즈 퍼블리시 완료")
            
        except Exception as e:
            self.get_logger().error(f"RViz2 퍼블리시 중 오류: {e}")

    def _publish_point_cloud(self, points_3d, topic_publisher, frame_id='map', color_rgb=None):
        """3D 포인트를 PointCloud2 메시지로 퍼블리시"""
        if points_3d.shape[0] == 0:
            return
            
        from sensor_msgs.msg import PointField
        from std_msgs.msg import Header
        import struct
            
        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = frame_id
        
        # PointCloud2 필드 정의
        fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
        ]
        
        # 색상 추가 (옵션)
        if color_rgb is not None:
            fields.extend([
                PointField(name='r', offset=12, datatype=PointField.UINT8, count=1),
                PointField(name='g', offset=13, datatype=PointField.UINT8, count=1),
                PointField(name='b', offset=14, datatype=PointField.UINT8, count=1),
            ])
            point_step = 15
        else:
            point_step = 12
        
        # 포인트 데이터 생성
        cloud_data = []
        for i, point in enumerate(points_3d):
            x, y, z = float(point[0]), float(point[1]), float(point[2])
            
            if color_rgb is not None:
                r, g, b = color_rgb if isinstance(color_rgb, tuple) else (255, 255, 255)
                cloud_data.append(struct.pack('fffBBB', x, y, z, r, g, b))
            else:
                cloud_data.append(struct.pack('fff', x, y, z))
        
        # PointCloud2 메시지 생성
        cloud_msg = PointCloud2()
        cloud_msg.header = header
        cloud_msg.height = 1
        cloud_msg.width = len(points_3d)
        cloud_msg.fields = fields
        cloud_msg.is_bigendian = False
        cloud_msg.point_step = point_step
        cloud_msg.row_step = point_step * len(points_3d)
        cloud_msg.data = b''.join(cloud_data)
        cloud_msg.is_dense = True
        
        topic_publisher.publish(cloud_msg)

    def _publish_keypoints_as_markers(self, keypoints_2d, topic_publisher, namespace, frame_id='map', color=(1.0, 0.0, 0.0, 1.0), scale=0.5):
        """2D 키포인트를 마커 배열로 퍼블리시"""
        if keypoints_2d.shape[0] == 0:
            return
            
        from visualization_msgs.msg import Marker
        
        marker_array = MarkerArray()
        
        for i, keypoint in enumerate(keypoints_2d):
            marker = Marker()
            marker.header.frame_id = frame_id
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = namespace
            marker.id = i
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            
            # 위치 설정 (2D 키포인트를 3D로 변환, z=0.5로 설정)
            marker.pose.position.x = float(keypoint[0])
            marker.pose.position.y = float(keypoint[1])
            marker.pose.position.z = 0.5
            marker.pose.orientation.w = 1.0
            
            # 크기 설정
            marker.scale.x = scale
            marker.scale.y = scale
            marker.scale.z = scale
            
            # 색상 설정
            marker.color.r = color[0]
            marker.color.g = color[1]
            marker.color.b = color[2]
            marker.color.a = color[3]
            
            # 수명 설정
            marker.lifetime = rclpy.duration.Duration(seconds=10.0).to_msg()
            
            marker_array.markers.append(marker)
        
        topic_publisher.publish(marker_array)

    def _apply_transform_to_points(self, points_3d, transform_result):
        """3D 포인트에 변환 적용"""
        if 'transform_matrix' in transform_result and transform_result['transform_matrix'] is not None:
            transform_matrix = transform_result['transform_matrix']
            # 동차 좌표로 변환
            points_homogeneous = np.hstack([points_3d, np.ones((points_3d.shape[0], 1))])
            # 변환 적용
            transformed_points = (transform_matrix @ points_homogeneous.T).T
            return transformed_points[:, :3]
        elif 'tx' in transform_result and 'ty' in transform_result and 'theta_deg' in transform_result:
            # 개별 변환 파라미터로 변환
            tx, ty, theta_deg = transform_result['tx'], transform_result['ty'], transform_result['theta_deg']
            theta_rad = np.deg2rad(theta_deg)
            cos_theta = np.cos(theta_rad)
            sin_theta = np.sin(theta_rad)
            
            # 2D 회전 및 평행이동 적용
            transformed_points = points_3d.copy()
            x_rot = cos_theta * points_3d[:, 0] - sin_theta * points_3d[:, 1] + tx
            y_rot = sin_theta * points_3d[:, 0] + cos_theta * points_3d[:, 1] + ty
            transformed_points[:, 0] = x_rot
            transformed_points[:, 1] = y_rot
            return transformed_points
        else:
            return points_3d

    def _apply_transform_to_keypoints(self, keypoints_2d, transform_result):
        """2D 키포인트에 변환 적용"""
        if 'tx' in transform_result and 'ty' in transform_result and 'theta_deg' in transform_result:
            tx, ty, theta_deg = transform_result['tx'], transform_result['ty'], transform_result['theta_deg']
            theta_rad = np.deg2rad(theta_deg)
            cos_theta = np.cos(theta_rad)
            sin_theta = np.sin(theta_rad)
            
            # 2D 회전 및 평행이동 적용
            x_rot = cos_theta * keypoints_2d[:, 0] - sin_theta * keypoints_2d[:, 1] + tx
            y_rot = sin_theta * keypoints_2d[:, 0] + cos_theta * keypoints_2d[:, 1] + ty
            return np.column_stack([x_rot, y_rot])
        else:
            return keypoints_2d


def main(args=None):
    rclpy.init(args=args)
    
    # multiprocessing을 위한 설정
    multiprocessing.freeze_support()
    
    node = FileProcessorNode()
    
    try:
        node.get_logger().info("파일 처리 노드 실행 중...")
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('키보드 인터럽트 수신, 종료 중...')
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main() 
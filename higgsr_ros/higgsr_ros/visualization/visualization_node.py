#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import numpy as np
import os

from sensor_msgs.msg import PointCloud2, PointField
from geometry_msgs.msg import TransformStamped, PoseStamped
from nav_msgs.msg import OccupancyGrid, Path
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import Header, ColorRGBA
from geometry_msgs.msg import Point, Quaternion, Pose
import tf_transformations
import struct

from higgsr_interface.msg import PointCloudInfo, Keypoints, DensityMap
from higgsr_ros.core import utils as core_utils
from higgsr_ros.core import feature_extraction as core_feature_extraction


class HiGGSRVisualizationNode(Node):
    def __init__(self):
        super().__init__('higgsr_visualization_node')
        self.get_logger().info("HiGGSR 시각화 노드 시작")

        # 파라미터 선언
        self.declare_parameter('map_frame_id', 'map')
        self.declare_parameter('base_frame_id', 'base_link')
        self.declare_parameter('visualization_scale', 1.0)
        self.declare_parameter('path_max_length', 100)
        self.declare_parameter('marker_lifetime', 10.0)
        
        # 디버깅 파라미터 추가
        self.declare_parameter('debug_use_identity_transform', False)
        self.declare_parameter('debug_log_transforms', True)
        
        # 글로벌 맵 관련 파라미터 추가
        self.declare_parameter('global_map_file_path', 'src/HiGGSR/Data/around_singong - Cloud.ply')
        self.declare_parameter('global_grid_size', 0.2)
        self.declare_parameter('global_min_points_for_density_calc', 3)
        self.declare_parameter('global_density_metric', 'std')
        self.declare_parameter('global_keypoint_density_threshold', 0.1)

        # 내부 상태 변수 - 단순화
        self.estimated_global_path = Path()  # 추정된 글로벌 위치들의 경로 (유일한 경로)
        self.estimated_global_path.header.frame_id = "map"
        
        # 글로벌 맵 데이터 저장 변수
        self.global_map_points_3d = None
        self.global_keypoints = None
        self.latest_live_scan = None
        self.latest_scan_keypoints = None
        self.latest_transform = None
        
        # 누적 이동량 추적용
        self.accumulated_dx = 0.0
        self.accumulated_dy = 0.0
        self.accumulated_dtheta = 0.0
        self.previous_transform = None
        
        # 구독자 생성
        self.pose_subscription = self.create_subscription(
            PoseStamped,
            'higgsr_pose',
            self.pose_callback,
            10)
        
        self.transform_subscription = self.create_subscription(
            TransformStamped,
            'higgsr_transform',
            self.transform_callback,
            10)
            
        # 라이다 토픽 구독 추가 (라이브 스캔 시각화용)
        self.declare_parameter('lidar_topic', '/ouster/points')
        self.declare_parameter('use_corrected_lidar_frame', True)  # os_sensor 프레임 사용 여부
        
        lidar_topic = self.get_parameter('lidar_topic').get_parameter_value().string_value
        use_corrected_frame = self.get_parameter('use_corrected_lidar_frame').get_parameter_value().bool_value
        
        if use_corrected_frame:
            self.get_logger().info(f"라이다 토픽에서 os_sensor 프레임으로 보정된 데이터 사용")
            # 여기서는 토픽 자체는 동일하게 구독하지만, 프레임 처리를 다르게 함
        
        self.lidar_subscription = self.create_subscription(
            PointCloud2,
            lidar_topic,
            self.lidar_callback,
            10)

        # 퍼블리셔 생성
        self.path_publisher = self.create_publisher(Path, 'higgsr_path', 10)
        self.estimated_path_publisher = self.create_publisher(Path, 'higgsr_estimated_path', 10)
        self.marker_publisher = self.create_publisher(MarkerArray, 'higgsr_markers', 10)
        self.pose_marker_publisher = self.create_publisher(Marker, 'higgsr_pose_marker', 10)
        self.trajectory_publisher = self.create_publisher(Marker, 'higgsr_trajectory', 10)
        self.estimated_trajectory_publisher = self.create_publisher(Marker, 'higgsr_estimated_trajectory', 10)
        
        # 포인트클라우드 퍼블리셔 추가
        self.global_map_publisher = self.create_publisher(PointCloud2, 'higgsr_global_map', 10)
        self.live_scan_publisher = self.create_publisher(PointCloud2, 'higgsr_live_scan', 10)
        self.original_scan_publisher = self.create_publisher(PointCloud2, 'higgsr_original_scan', 10)
        
        # 키포인트 퍼블리셔 추가
        self.global_keypoints_publisher = self.create_publisher(MarkerArray, 'higgsr_global_keypoints', 10)
        self.scan_keypoints_publisher = self.create_publisher(MarkerArray, 'higgsr_scan_keypoints', 10)
        self.original_keypoints_publisher = self.create_publisher(MarkerArray, 'higgsr_original_keypoints', 10)
        
        # 원래 포즈 퍼블리셔 추가
        self.original_pose_publisher = self.create_publisher(PoseStamped, 'higgsr_original_pose', 10)

        # 글로벌 맵 자동 로드
        self._load_global_map()

        # 타이머 생성 (주기적 시각화 업데이트)
        self.visualization_timer = self.create_timer(0.1, self.update_visualization)
        
        self.get_logger().info("시각화 노드 초기화 완료")

    def _load_global_map(self):
        """글로벌 맵을 자동으로 로드하고 키포인트 추출"""
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
            
            # 글로벌 키포인트 추출
            self._extract_global_keypoints()
            
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
    
    def _extract_global_keypoints(self):
        """글로벌 맵에서 키포인트 추출"""
        try:
            if self.global_map_points_3d is None:
                return
                
            # 파라미터 가져오기
            grid_size = self.get_parameter('global_grid_size').get_parameter_value().double_value
            min_points = self.get_parameter('global_min_points_for_density_calc').get_parameter_value().integer_value
            density_metric = self.get_parameter('global_density_metric').get_parameter_value().string_value
            keypoint_threshold = self.get_parameter('global_keypoint_density_threshold').get_parameter_value().double_value

            # 1. Pillar Map 생성
            density_map_global, x_edges_global, y_edges_global = \
                core_utils.create_2d_height_variance_map(
                    self.global_map_points_3d,
                    grid_cell_size=grid_size,
                    min_points_per_cell=min_points,
                    density_metric=density_metric)

            if density_map_global.size == 0:
                raise ValueError("글로벌 Pillar Map 생성 실패")

            self.get_logger().info(f"글로벌 Pillar Map 생성 완료: {density_map_global.shape}")

            # 2. 키포인트 추출
            self.global_keypoints = core_feature_extraction.extract_high_density_keypoints(
                density_map_global,
                x_edges_global,
                y_edges_global,
                density_threshold=keypoint_threshold)

            self.get_logger().info(f"글로벌 키포인트 추출 완료: {self.global_keypoints.shape[0]} 개")

        except Exception as e:
            self.get_logger().error(f"글로벌 키포인트 추출 중 오류: {e}")

    def lidar_callback(self, msg: PointCloud2):
        """라이다 데이터 수신 콜백 - 라이브 스캔 시각화용"""
        try:
            self.latest_live_scan = msg
            # 실시간으로 키포인트 추출하지 않고, 변환 완료 시에만 처리
            
        except Exception as e:
            self.get_logger().error(f"라이다 콜백 처리 중 오류: {e}")

    def pose_callback(self, msg: PoseStamped):
        """포즈 메시지 수신 콜백"""
        try:
            # 경로에 포즈 추가
            self.add_pose_to_path(msg)
            
            # 포즈 마커 생성
            self.publish_pose_marker(msg)
            
        except Exception as e:
            self.get_logger().error(f"포즈 콜백 처리 중 오류: {e}")

    def transform_callback(self, msg: TransformStamped):
        """변환 메시지 수신 콜백"""
        try:
            # 변환 정보 저장
            self.latest_transform = msg
            
            # 디버깅 모드 확인
            debug_log = self.get_parameter('debug_log_transforms').get_parameter_value().bool_value
            
            if debug_log:
                self.get_logger().info(f"변환 수신: tx={msg.transform.translation.x:.3f}, ty={msg.transform.translation.y:.3f}")
                quat = [msg.transform.rotation.x, msg.transform.rotation.y, msg.transform.rotation.z, msg.transform.rotation.w]
                euler = tf_transformations.euler_from_quaternion(quat)
                theta_deg = np.rad2deg(euler[2])
                self.get_logger().info(f"회전각: {theta_deg:.1f}도")
            
            # 1. 추정된 글로벌 위치 (변환된 포즈) - Open3D와 동일한 접근법
            estimated_global_pose = PoseStamped()
            estimated_global_pose.header.frame_id = "map"  # 모든 것을 map 프레임으로 통일
            estimated_global_pose.header.stamp = msg.header.stamp
            estimated_global_pose.pose.position.x = msg.transform.translation.x
            estimated_global_pose.pose.position.y = msg.transform.translation.y
            estimated_global_pose.pose.position.z = msg.transform.translation.z
            estimated_global_pose.pose.orientation = msg.transform.rotation
            
            # 추정된 글로벌 경로에 추가 (이것만 유지)
            self.estimated_global_path.header.frame_id = "map"
            self.estimated_global_path.header.stamp = estimated_global_pose.header.stamp
            self.estimated_global_path.poses.append(estimated_global_pose)
            max_length = self.get_parameter('path_max_length').get_parameter_value().integer_value
            if len(self.estimated_global_path.poses) > max_length:
                self.estimated_global_path.poses = self.estimated_global_path.poses[-max_length:]
            
            # 2. 원래 포즈 퍼블리시 (map 프레임에서 원점) - Open3D에서 변환 전 스캔과 같은 개념
            original_pose_msg = PoseStamped()
            original_pose_msg.header.frame_id = "map"  # map 프레임으로 통일
            original_pose_msg.header.stamp = msg.header.stamp
            original_pose_msg.pose.position.x = 0.0
            original_pose_msg.pose.position.y = 0.0
            original_pose_msg.pose.position.z = 0.0
            original_pose_msg.pose.orientation.w = 1.0
            
            self.original_pose_publisher.publish(original_pose_msg)
            
            # 3. 추정된 글로벌 포즈 콜백 호출 (시각화용)
            self.pose_callback(estimated_global_pose)
            
            # 라이브 스캔이 있으면 키포인트 추출 및 시각화 업데이트
            if self.latest_live_scan is not None:
                if debug_log:
                    self.get_logger().info("라이브 스캔에 변환 적용 중... (Open3D와 동일한 방식)")
                self._extract_live_scan_keypoints()
                self._publish_all_visualization_data()
            else:
                if debug_log:
                    self.get_logger().warn("라이브 스캔 데이터가 없습니다.")
            
        except Exception as e:
            self.get_logger().error(f"변환 콜백 처리 중 오류: {e}")
            import traceback
            self.get_logger().error(f"상세 오류:\n{traceback.format_exc()}")

    def _extract_live_scan_keypoints(self):
        """최신 라이브 스캔에서 키포인트 추출"""
        try:
            if self.latest_live_scan is None:
                return
                
            # ROS PointCloud2를 numpy로 변환
            from higgsr_ros.utils import ros_utils
            live_scan_points = ros_utils.convert_ros_point_cloud2_to_numpy(
                self.latest_live_scan, field_names=('x', 'y', 'z'))

            if live_scan_points is None or live_scan_points.shape[0] == 0:
                self.get_logger().warn("라이브 스캔 변환 실패 또는 빈 포인트클라우드")
                return

            # 라이브 스캔 처리 파라미터 (글로벌 맵과 동일한 파라미터 사용)
            grid_size = self.get_parameter('global_grid_size').get_parameter_value().double_value
            min_points = self.get_parameter('global_min_points_for_density_calc').get_parameter_value().integer_value
            density_metric = self.get_parameter('global_density_metric').get_parameter_value().string_value
            keypoint_threshold = self.get_parameter('global_keypoint_density_threshold').get_parameter_value().double_value

            # 라이브 스캔 Pillar Map 생성
            density_map_scan, x_edges_scan, y_edges_scan = \
                core_utils.create_2d_height_variance_map(
                    live_scan_points,
                    grid_cell_size=grid_size,
                    min_points_per_cell=min_points,
                    density_metric=density_metric)

            if density_map_scan.size == 0:
                self.get_logger().warn("라이브 스캔 Pillar Map 생성 실패")
                return

            # 라이브 스캔 키포인트 추출
            self.latest_scan_keypoints = core_feature_extraction.extract_high_density_keypoints(
                density_map_scan, x_edges_scan, y_edges_scan,
                density_threshold=keypoint_threshold)

            self.get_logger().info(f"라이브 스캔 키포인트 추출 완료: {self.latest_scan_keypoints.shape[0]} 개")

        except Exception as e:
            self.get_logger().error(f"라이브 스캔 키포인트 추출 중 오류: {e}")

    def _publish_all_visualization_data(self):
        """모든 시각화 데이터를 퍼블리시 - Open3D와 동일한 방식"""
        try:
            frame_id = "map"  # 모든 것을 map 프레임으로 통일
            
            self.get_logger().info("=== RViz 시각화 데이터 퍼블리시 시작 ===")
            
            # 1. 글로벌 맵 포인트클라우드 퍼블리시 (흰색) - map 프레임에서 (0,0,0) 기준
            if self.global_map_points_3d is not None and self.global_map_points_3d.shape[0] > 0:
                self.get_logger().info(f"글로벌 맵 퍼블리시: {self.global_map_points_3d.shape[0]} 포인트 (map 프레임 기준)")
                self.publish_point_cloud(self.global_map_points_3d, self.global_map_publisher, frame_id, (255, 255, 255))
                
            # 2. 라이브 스캔 처리 - Open3D와 동일한 방식
            if self.latest_live_scan is not None:
                from higgsr_ros.utils import ros_utils
                live_scan_points = ros_utils.convert_ros_point_cloud2_to_numpy(
                    self.latest_live_scan, field_names=('x', 'y', 'z'))
                    
                if live_scan_points is not None and live_scan_points.shape[0] > 0:
                    self.get_logger().info(f"라이브 스캔 처리: {live_scan_points.shape[0]} 포인트")
                    
                    # 원래 스캔 퍼블리시 (노란색) - map 프레임에서 (0,0,0) 기준 (Open3D 변환 전과 동일)
                    self.get_logger().info("원본 라이브 스캔 퍼블리시 (노란색, map 프레임 기준)")
                    sample_idx = min(3, live_scan_points.shape[0])
                    self.get_logger().info(f"원본 스캔 샘플 (첫 {sample_idx}개):")
                    for i in range(sample_idx):
                        point = live_scan_points[i]
                        self.get_logger().info(f"  원본[{i}]: ({point[0]:.3f}, {point[1]:.3f}, {point[2]:.3f})")
                    
                    self.publish_point_cloud(live_scan_points, self.original_scan_publisher, frame_id, (255, 255, 0))
                    
                    # 변환된 스캔 퍼블리시 (빨간색) - Open3D에서 변환 후와 동일
                    if self.latest_transform is not None:
                        self.get_logger().info("=== RViz 포인트클라우드 변환 적용 ===")
                        transform_dict = self._transform_to_dict(self.latest_transform)
                        self.get_logger().info(f"RViz 변환에 사용할 파라미터: {transform_dict}")
                        
                        transformed_scan = self.apply_transform_to_points_open3d_style(live_scan_points, transform_dict)
                        
                        # 변환 결과 샘플 로깅
                        sample_idx = min(3, live_scan_points.shape[0])
                        self.get_logger().info(f"RViz 포인트 변환 결과 샘플 (첫 {sample_idx}개):")
                        for i in range(sample_idx):
                            orig = live_scan_points[i]
                            trans = transformed_scan[i]
                            dx, dy, dz = trans[0] - orig[0], trans[1] - orig[1], trans[2] - orig[2]
                            self.get_logger().info(f"  RViz[{i}]: ({orig[0]:.3f}, {orig[1]:.3f}, {orig[2]:.3f}) → ({trans[0]:.3f}, {trans[1]:.3f}, {trans[2]:.3f}) [delta: ({dx:.3f}, {dy:.3f}, {dz:.3f})]")
                        
                        self.get_logger().info("변환된 라이브 스캔 퍼블리시 (빨간색, map 프레임 기준)")
                        self.publish_point_cloud(transformed_scan, self.live_scan_publisher, frame_id, (255, 0, 0))
                    else:
                        self.get_logger().warn("변환 정보가 없어서 원본 스캔을 그대로 퍼블리시")
                        self.publish_point_cloud(live_scan_points, self.live_scan_publisher, frame_id, (255, 0, 0))
                        
            # 3. 글로벌 키포인트 퍼블리시 (파란색)
            if self.global_keypoints is not None and self.global_keypoints.shape[0] > 0:
                self.get_logger().info(f"글로벌 키포인트 퍼블리시: {self.global_keypoints.shape[0]}개 (파란색)")
                self.publish_keypoints_as_markers(self.global_keypoints, self.global_keypoints_publisher, 
                                                "global_keypoints", frame_id, (0.0, 0.0, 1.0, 1.0), 0.8)
                                                
            # 4. 스캔 키포인트 퍼블리시 (원래 + 변환된)
            if self.latest_scan_keypoints is not None and self.latest_scan_keypoints.shape[0] > 0:
                self.get_logger().info(f"스캔 키포인트 처리: {self.latest_scan_keypoints.shape[0]}개")
                
                # 원래 키포인트 퍼블리시 (주황색) - map 프레임에서 (0,0,0) 기준
                self.get_logger().info("원본 스캔 키포인트 퍼블리시 (주황색)")
                sample_idx = min(3, self.latest_scan_keypoints.shape[0])
                self.get_logger().info(f"원본 키포인트 샘플 (첫 {sample_idx}개):")
                for i in range(sample_idx):
                    kp = self.latest_scan_keypoints[i]
                    self.get_logger().info(f"  원본키포인트[{i}]: ({kp[0]:.3f}, {kp[1]:.3f})")
                
                self.publish_keypoints_as_markers(self.latest_scan_keypoints, self.original_keypoints_publisher, 
                                                "original_keypoints", frame_id, (1.0, 0.5, 0.0, 1.0), 0.4)
                
                # 변환된 키포인트 퍼블리시 (녹색) - Open3D 변환 후와 동일
                if self.latest_transform is not None:
                    self.get_logger().info("=== RViz 키포인트 변환 적용 ===")
                    transform_dict = self._transform_to_dict(self.latest_transform)
                    transformed_keypoints = self.apply_transform_to_keypoints_open3d_style(self.latest_scan_keypoints, transform_dict)
                    
                    # 키포인트 변환 결과 샘플 로깅
                    sample_idx = min(3, self.latest_scan_keypoints.shape[0])
                    self.get_logger().info(f"RViz 키포인트 변환 결과 샘플 (첫 {sample_idx}개):")
                    for i in range(sample_idx):
                        orig = self.latest_scan_keypoints[i]
                        trans = transformed_keypoints[i]
                        dx, dy = trans[0] - orig[0], trans[1] - orig[1]
                        self.get_logger().info(f"  RViz키포인트[{i}]: ({orig[0]:.3f}, {orig[1]:.3f}) → ({trans[0]:.3f}, {trans[1]:.3f}) [delta: ({dx:.3f}, {dy:.3f})]")
                    
                    self.get_logger().info("변환된 스캔 키포인트 퍼블리시 (녹색)")
                    self.publish_keypoints_as_markers(transformed_keypoints, self.scan_keypoints_publisher, 
                                                    "scan_keypoints", frame_id, (0.0, 1.0, 0.0, 1.0), 0.6)
                else:
                    self.get_logger().warn("변환 정보가 없어서 원본 키포인트를 그대로 퍼블리시")
                    self.publish_keypoints_as_markers(self.latest_scan_keypoints, self.scan_keypoints_publisher, 
                                                    "scan_keypoints", frame_id, (0.0, 1.0, 0.0, 1.0), 0.6)
                    
            self.get_logger().info("=== RViz 시각화 데이터 퍼블리시 완료 ===")
            
        except Exception as e:
            self.get_logger().error(f"시각화 데이터 퍼블리시 중 오류: {e}")
            import traceback
            self.get_logger().error(f"상세 오류:\n{traceback.format_exc()}")

    def _transform_to_dict(self, transform_stamped):
        """TransformStamped를 딕셔너리로 변환"""
        try:
            # 디버깅 모드 확인
            use_identity = self.get_parameter('debug_use_identity_transform').get_parameter_value().bool_value
            debug_log = self.get_parameter('debug_log_transforms').get_parameter_value().bool_value
            
            if use_identity:
                # 아이덴티티 변환 (변환 없음)
                if debug_log:
                    self.get_logger().info("디버깅 모드: 아이덴티티 변환 사용")
                return {
                    'tx': 0.0,
                    'ty': 0.0,
                    'theta_deg': 0.0,
                    'transform_matrix': np.eye(4)
                }
            
            # 쿼터니언 추출
            quat = [
                transform_stamped.transform.rotation.x,
                transform_stamped.transform.rotation.y,
                transform_stamped.transform.rotation.z,
                transform_stamped.transform.rotation.w
            ]
            
            # 번역 추출
            tx = transform_stamped.transform.translation.x
            ty = transform_stamped.transform.translation.y
            
            if debug_log:
                self.get_logger().info(f"=== 회전 변환 디버깅 ===")
                self.get_logger().info(f"원본 쿼터니언: x={quat[0]:.6f}, y={quat[1]:.6f}, z={quat[2]:.6f}, w={quat[3]:.6f}")
                
            # 쿼터니언을 오일러 각도로 변환
            euler = tf_transformations.euler_from_quaternion(quat)
            roll, pitch, yaw = euler[0], euler[1], euler[2]
            theta_deg = np.rad2deg(yaw)  # z축 회전각 (yaw)
            
            if debug_log:
                self.get_logger().info(f"오일러 각도 (라디안): roll={roll:.6f}, pitch={pitch:.6f}, yaw={yaw:.6f}")
                self.get_logger().info(f"오일러 각도 (도): roll={np.rad2deg(roll):.3f}°, pitch={np.rad2deg(pitch):.3f}°, yaw={theta_deg:.3f}°")
                
                # 역변환 테스트
                test_quat = tf_transformations.quaternion_from_euler(roll, pitch, yaw)
                self.get_logger().info(f"역변환 쿼터니언: x={test_quat[0]:.6f}, y={test_quat[1]:.6f}, z={test_quat[2]:.6f}, w={test_quat[3]:.6f}")
                
                # 순수 yaw 쿼터니언 생성 테스트 (자동차 움직임에 적합)
                pure_yaw_quat = tf_transformations.quaternion_from_euler(0, 0, yaw)
                self.get_logger().info(f"순수 yaw 쿼터니언: x={pure_yaw_quat[0]:.6f}, y={pure_yaw_quat[1]:.6f}, z={pure_yaw_quat[2]:.6f}, w={pure_yaw_quat[3]:.6f}")
            
            # 4x4 변환 행렬 생성
            transform_matrix_4x4 = core_utils.create_transform_matrix_4x4(tx, ty, theta_deg)
            
            if debug_log:
                self.get_logger().info(f"변환 파라미터: tx={tx:.3f}, ty={ty:.3f}, theta={theta_deg:.1f}°")
                self.get_logger().info(f"4x4 변환 행렬:\n{transform_matrix_4x4}")
                
                # 변환 행렬에서 회전 부분 확인
                rotation_part = transform_matrix_4x4[:3, :3]
                self.get_logger().info(f"변환 행렬의 회전 부분:\n{rotation_part}")
                
                # 변환 행렬에서 각도 역계산
                calculated_angle = np.arctan2(transform_matrix_4x4[1, 0], transform_matrix_4x4[0, 0])
                calculated_angle_deg = np.rad2deg(calculated_angle)
                self.get_logger().info(f"변환 행렬에서 역계산된 각도: {calculated_angle_deg:.3f}°")
                
                self.get_logger().info(f"=== 회전 변환 디버깅 완료 ===")
            
            return {
                'tx': tx,
                'ty': ty,
                'theta_deg': theta_deg,
                'transform_matrix': transform_matrix_4x4
            }
        except Exception as e:
            self.get_logger().error(f"변환 딕셔너리 생성 중 오류: {e}")
            return {'tx': 0.0, 'ty': 0.0, 'theta_deg': 0.0, 'transform_matrix': None}

    def add_pose_to_path(self, pose_msg: PoseStamped):
        """경로에 새 포즈 추가"""
        self.estimated_global_path.header.stamp = pose_msg.header.stamp
        self.estimated_global_path.poses.append(pose_msg)
        
        # 경로 길이 제한
        max_length = self.get_parameter('path_max_length').get_parameter_value().integer_value
        if len(self.estimated_global_path.poses) > max_length:
            self.estimated_global_path.poses = self.estimated_global_path.poses[-max_length:]

    def publish_pose_marker(self, pose_msg: PoseStamped):
        """현재 포즈를 마커로 퍼블리시"""
        marker = Marker()
        marker.header = pose_msg.header
        marker.ns = "higgsr_pose"
        marker.id = 0
        marker.type = Marker.ARROW
        marker.action = Marker.ADD
        
        # 포즈 설정
        marker.pose = pose_msg.pose
        
        # 크기 설정
        scale = self.get_parameter('visualization_scale').get_parameter_value().double_value
        marker.scale.x = 2.0 * scale
        marker.scale.y = 0.3 * scale
        marker.scale.z = 0.3 * scale
        
        # 색상 설정 (빨간색)
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.color.a = 1.0
        
        # 수명 설정
        lifetime = self.get_parameter('marker_lifetime').get_parameter_value().double_value
        marker.lifetime = rclpy.duration.Duration(seconds=lifetime).to_msg()
        
        self.pose_marker_publisher.publish(marker)

    def publish_trajectory_marker(self):
        """로봇 실제 이동 궤적을 라인 마커로 퍼블리시"""
        if len(self.estimated_global_path.poses) < 2:
            return
            
        marker = Marker()
        marker.header = self.estimated_global_path.header
        marker.ns = "higgsr_trajectory"
        marker.id = 1
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        
        # 포인트 추가
        for pose in self.estimated_global_path.poses:
            point = Point()
            point.x = pose.pose.position.x
            point.y = pose.pose.position.y
            point.z = pose.pose.position.z + 0.1  # 약간 위에 표시
            marker.points.append(point)
        
        # 라인 스타일 설정
        scale = self.get_parameter('visualization_scale').get_parameter_value().double_value
        marker.scale.x = 0.1 * scale
        
        # 색상 설정 (파란색 - 로봇의 실제 이동)
        marker.color.r = 0.0
        marker.color.g = 0.0
        marker.color.b = 1.0
        marker.color.a = 1.0
        
        self.trajectory_publisher.publish(marker)

    def publish_estimated_trajectory_marker(self):
        """추정된 글로벌 위치 궤적을 라인 마커로 퍼블리시"""
        if len(self.estimated_global_path.poses) < 2:
            return
            
        marker = Marker()
        marker.header = self.estimated_global_path.header
        marker.ns = "higgsr_estimated_trajectory"
        marker.id = 2
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        
        # 포인트 추가
        for pose in self.estimated_global_path.poses:
            point = Point()
            point.x = pose.pose.position.x
            point.y = pose.pose.position.y
            point.z = pose.pose.position.z + 0.2  # 약간 더 위에 표시
            marker.points.append(point)
        
        # 라인 스타일 설정
        scale = self.get_parameter('visualization_scale').get_parameter_value().double_value
        marker.scale.x = 0.15 * scale  # 약간 더 굵게
        
        # 색상 설정 (보라색 - 추정된 글로벌 위치)
        marker.color.r = 0.8
        marker.color.g = 0.0
        marker.color.b = 0.8
        marker.color.a = 1.0
        
        self.estimated_trajectory_publisher.publish(marker)

    def publish_statistics_markers(self):
        """통계 정보를 텍스트 마커로 퍼블리시"""
        marker_array = MarkerArray()
        
        # 경로 길이 표시
        if len(self.estimated_global_path.poses) > 1:
            total_distance = self.calculate_path_distance()
            
            text_marker = Marker()
            text_marker.header.frame_id = self.get_parameter('map_frame_id').get_parameter_value().string_value
            text_marker.header.stamp = self.get_clock().now().to_msg()
            text_marker.ns = "higgsr_stats"
            text_marker.id = 2
            text_marker.type = Marker.TEXT_VIEW_FACING
            text_marker.action = Marker.ADD
            
            # 텍스트 내용
            text_marker.text = f"Total Distance: {total_distance:.2f}m\nPoses: {len(self.estimated_global_path.poses)}"
            
            # 위치 (맵 원점 근처)
            text_marker.pose.position.x = 0.0
            text_marker.pose.position.y = 0.0
            text_marker.pose.position.z = 5.0
            text_marker.pose.orientation.w = 1.0
            
            # 크기와 색상
            scale = self.get_parameter('visualization_scale').get_parameter_value().double_value
            text_marker.scale.z = 1.0 * scale
            text_marker.color.r = 1.0
            text_marker.color.g = 1.0
            text_marker.color.b = 1.0
            text_marker.color.a = 1.0
            
            marker_array.markers.append(text_marker)
        
        if marker_array.markers:
            self.marker_publisher.publish(marker_array)

    def calculate_path_distance(self):
        """경로의 총 거리 계산"""
        total_distance = 0.0
        for i in range(1, len(self.estimated_global_path.poses)):
            prev_pose = self.estimated_global_path.poses[i-1].pose.position
            curr_pose = self.estimated_global_path.poses[i].pose.position
            
            dx = curr_pose.x - prev_pose.x
            dy = curr_pose.y - prev_pose.y
            dz = curr_pose.z - prev_pose.z
            
            distance = np.sqrt(dx*dx + dy*dy + dz*dz)
            total_distance += distance
            
        return total_distance

    def publish_point_cloud(self, points_3d, topic_publisher, frame_id='map', color_rgb=None):
        """3D 포인트를 PointCloud2 메시지로 퍼블리시"""
        if points_3d.shape[0] == 0:
            return
            
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

    def publish_keypoints_as_markers(self, keypoints_2d, topic_publisher, namespace, frame_id='map', color=(1.0, 0.0, 0.0, 1.0), scale=0.5):
        """2D 키포인트를 마커 배열로 퍼블리시"""
        if keypoints_2d.shape[0] == 0:
            return
            
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
            viz_scale = self.get_parameter('visualization_scale').get_parameter_value().double_value
            marker.scale.x = scale * viz_scale
            marker.scale.y = scale * viz_scale
            marker.scale.z = scale * viz_scale
            
            # 색상 설정
            marker.color.r = color[0]
            marker.color.g = color[1]
            marker.color.b = color[2]
            marker.color.a = color[3]
            
            # 수명 설정
            lifetime = self.get_parameter('marker_lifetime').get_parameter_value().double_value
            marker.lifetime = rclpy.duration.Duration(seconds=lifetime).to_msg()
            
            marker_array.markers.append(marker)
        
        topic_publisher.publish(marker_array)

    def update_visualization(self):
        """주기적 시각화 업데이트 - 단순화된 Open3D 방식"""
        try:
            # 추정된 글로벌 경로 퍼블리시 (변환된 위치들의 연결)
            if self.estimated_global_path.poses:
                self.estimated_path_publisher.publish(self.estimated_global_path)
                
            # 추정된 글로벌 위치 궤적 마커 퍼블리시
            self.publish_estimated_trajectory_marker()
            
            # 통계 마커 퍼블리시
            self.publish_statistics_markers()
            
            # 글로벌 맵 주기적 퍼블리시 (변환이 없을 때도 보이도록)
            if self.global_map_points_3d is not None and self.global_map_points_3d.shape[0] > 0:
                self.publish_point_cloud(self.global_map_points_3d, self.global_map_publisher, "map", (255, 255, 255))
                
            # 글로벌 키포인트 주기적 퍼블리시
            if self.global_keypoints is not None and self.global_keypoints.shape[0] > 0:
                self.publish_keypoints_as_markers(self.global_keypoints, self.global_keypoints_publisher, 
                                                "global_keypoints", "map", (0.0, 0.0, 1.0, 1.0), 0.8)
            
        except Exception as e:
            self.get_logger().error(f"시각화 업데이트 중 오류: {e}")

    def apply_transform_to_points(self, points_3d, transform_result):
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

    def apply_transform_to_keypoints(self, keypoints_2d, transform_result):
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

    def apply_transform_to_points_open3d_style(self, points_3d, transform_result):
        """Open3D와 동일한 방식으로 3D 포인트에 4x4 변환 행렬 적용"""
        try:
            debug_log = self.get_parameter('debug_log_transforms').get_parameter_value().bool_value
            
            if 'transform_matrix' in transform_result and transform_result['transform_matrix'] is not None:
                transform_matrix = transform_result['transform_matrix']
                if debug_log:
                    self.get_logger().info(f"4x4 변환 행렬 사용:\n{transform_matrix}")
                
                # 동차 좌표로 변환 (x, y, z, 1)
                points_homogeneous = np.hstack([points_3d, np.ones((points_3d.shape[0], 1))])
                
                # 4x4 변환 행렬 적용 (Open3D와 동일한 방식)
                transformed_points_homogeneous = (transform_matrix @ points_homogeneous.T).T
                
                # 3D 좌표만 반환
                transformed_points = transformed_points_homogeneous[:, :3]
                
                # 변환 결과 확인 로그
                if debug_log:
                    sample_idx = min(5, points_3d.shape[0])
                    self.get_logger().info(f"변환 전 샘플 포인트: {points_3d[:sample_idx]}")
                    self.get_logger().info(f"변환 후 샘플 포인트: {transformed_points[:sample_idx]}")
                
                return transformed_points
            else:
                if debug_log:
                    self.get_logger().warn("4x4 변환 행렬이 없어서 원본 포인트 반환")
                return points_3d
                
        except Exception as e:
            self.get_logger().error(f"Open3D 스타일 포인트 변환 중 오류: {e}")
            return points_3d

    def apply_transform_to_keypoints_open3d_style(self, keypoints_2d, transform_result):
        """Open3D와 동일한 방식으로 2D 키포인트에 변환 적용"""
        try:
            if 'transform_matrix' in transform_result and transform_result['transform_matrix'] is not None:
                transform_matrix = transform_result['transform_matrix']
                
                # 2D 키포인트를 3D로 확장 (z=0)
                keypoints_3d = np.hstack([keypoints_2d, np.zeros((keypoints_2d.shape[0], 1))])
                
                # 동차 좌표로 변환
                keypoints_homogeneous = np.hstack([keypoints_3d, np.ones((keypoints_3d.shape[0], 1))])
                
                # 4x4 변환 행렬 적용
                transformed_keypoints_homogeneous = (transform_matrix @ keypoints_homogeneous.T).T
                
                # 2D 좌표만 반환 (x, y)
                transformed_keypoints = transformed_keypoints_homogeneous[:, :2]
                
                return transformed_keypoints
            else:
                # fallback to original method
                return self.apply_transform_to_keypoints(keypoints_2d, transform_result)
                
        except Exception as e:
            self.get_logger().error(f"Open3D 스타일 키포인트 변환 중 오류: {e}")
            return keypoints_2d

    def clear_visualization(self):
        """시각화 초기화 - 단순화"""
        self.estimated_global_path.poses.clear()
        
        # 마커 삭제
        delete_marker = Marker()
        delete_marker.action = Marker.DELETEALL
        self.pose_marker_publisher.publish(delete_marker)
        self.estimated_trajectory_publisher.publish(delete_marker)
        
        marker_array = MarkerArray()
        marker_array.markers.append(delete_marker)
        self.marker_publisher.publish(marker_array)
        
        self.get_logger().info("시각화 초기화 완료 (단순화된 방식)")


def main(args=None):
    rclpy.init(args=args)
    
    node = HiGGSRVisualizationNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('키보드 인터럽트 수신, 종료 중...')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main() 
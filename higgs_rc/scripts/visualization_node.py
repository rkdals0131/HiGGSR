#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import numpy as np
import os

from sensor_msgs.msg import PointCloud2, PointField
from geometry_msgs.msg import TransformStamped, PoseStamped
from nav_msgs.msg import Path
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import Header
import tf_transformations
import struct

from higgsr_ros.core import utils as core_utils
from higgsr_ros.core import feature_extraction as core_feature_extraction


class HiGGSRVisualizationNode(Node):
    """
    HiGGSR 시각화 노드 (file_processor 방식에 최적화)
    - 완벽하게 작동하는 file_processor + test_rviz 시스템에 맞게 간소화
    - 핵심 시각화 기능만 유지
    """
    
    def __init__(self):
        super().__init__('higgsr_visualization_node')
        self.get_logger().info("HiGGSR 시각화 노드 시작 (file_processor 최적화)")

        # 파라미터 선언 (간소화)
        self.declare_parameter('map_frame_id', 'map')
        self.declare_parameter('visualization_scale', 1.0)
        self.declare_parameter('path_max_length', 100)
        self.declare_parameter('marker_lifetime', 10.0)

        # 내부 상태 변수 (간소화)
        self.estimated_path = Path()
        self.estimated_path.header.frame_id = "map"

        # 구독자 생성 (핵심만 유지)
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

        # 퍼블리셔 생성 (핵심만 유지)
        self.path_publisher = self.create_publisher(Path, 'higgsr_estimated_path', 10)
        self.pose_marker_publisher = self.create_publisher(Marker, 'higgsr_pose_marker', 10)
        self.trajectory_publisher = self.create_publisher(Marker, 'higgsr_trajectory', 10)
        
        # 통계 마커 퍼블리셔
        self.marker_publisher = self.create_publisher(MarkerArray, 'higgsr_stats_markers', 10)

        # 타이머 생성 (주기적 시각화 업데이트)
        self.visualization_timer = self.create_timer(0.1, self.update_visualization)
        
        self.get_logger().info("시각화 노드 초기화 완료 (간소화된 방식)")
        self.get_logger().info("file_processor + test_rviz 시스템과 완벽 호환")

    def pose_callback(self, msg: PoseStamped):
        """포즈 메시지 수신 콜백 (간소화)"""
        try:
            # 경로에 포즈 추가
            self.add_pose_to_path(msg)
            
            # 포즈 마커 생성
            self.publish_pose_marker(msg)
            
        except Exception as e:
            self.get_logger().error(f"포즈 콜백 처리 중 오류: {e}")

    def transform_callback(self, msg: TransformStamped):
        """변환 메시지 수신 콜백 (간소화)"""
        try:
            # 추정된 글로벌 위치로 변환
            estimated_pose = PoseStamped()
            estimated_pose.header.frame_id = "map"
            estimated_pose.header.stamp = msg.header.stamp
            estimated_pose.pose.position.x = msg.transform.translation.x
            estimated_pose.pose.position.y = msg.transform.translation.y
            estimated_pose.pose.position.z = msg.transform.translation.z
            estimated_pose.pose.orientation = msg.transform.rotation
            
            # 경로에 추가
            self.pose_callback(estimated_pose)
            
        except Exception as e:
            self.get_logger().error(f"변환 콜백 처리 중 오류: {e}")

    def add_pose_to_path(self, pose_msg: PoseStamped):
        """경로에 새 포즈 추가 (간소화)"""
        self.estimated_path.header.stamp = pose_msg.header.stamp
        self.estimated_path.poses.append(pose_msg)
        
        # 경로 길이 제한
        max_length = self.get_parameter('path_max_length').get_parameter_value().integer_value
        if len(self.estimated_path.poses) > max_length:
            self.estimated_path.poses = self.estimated_path.poses[-max_length:]

    def publish_pose_marker(self, pose_msg: PoseStamped):
        """현재 포즈를 마커로 퍼블리시 (간소화)"""
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
        """궤적을 라인 마커로 퍼블리시 (간소화)"""
        if len(self.estimated_path.poses) < 2:
            return
            
        marker = Marker()
        marker.header = self.estimated_path.header
        marker.ns = "higgsr_trajectory"
        marker.id = 1
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        
        # 포인트 추가
        for pose in self.estimated_path.poses:
            from geometry_msgs.msg import Point
            point = Point()
            point.x = pose.pose.position.x
            point.y = pose.pose.position.y
            point.z = pose.pose.position.z + 0.1  # 약간 위에 표시
            marker.points.append(point)
        
        # 라인 스타일 설정
        scale = self.get_parameter('visualization_scale').get_parameter_value().double_value
        marker.scale.x = 0.1 * scale
        
        # 색상 설정 (보라색)
        marker.color.r = 0.8
        marker.color.g = 0.0
        marker.color.b = 0.8
        marker.color.a = 1.0
        
        self.trajectory_publisher.publish(marker)

    def publish_statistics_markers(self):
        """통계 정보를 텍스트 마커로 퍼블리시 (간소화)"""
        if len(self.estimated_path.poses) > 1:
            marker_array = MarkerArray()
            total_distance = self.calculate_path_distance()
            
            from visualization_msgs.msg import Marker
            text_marker = Marker()
            text_marker.header.frame_id = self.get_parameter('map_frame_id').get_parameter_value().string_value
            text_marker.header.stamp = self.get_clock().now().to_msg()
            text_marker.ns = "higgsr_stats"
            text_marker.id = 0
            text_marker.type = Marker.TEXT_VIEW_FACING
            text_marker.action = Marker.ADD
            
            # 텍스트 내용
            text_marker.text = f"HiGGSR Results\nDistance: {total_distance:.2f}m\nPoses: {len(self.estimated_path.poses)}\n(file_processor style)"
            
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
            self.marker_publisher.publish(marker_array)

    def calculate_path_distance(self):
        """경로의 총 거리 계산 (간소화)"""
        total_distance = 0.0
        for i in range(1, len(self.estimated_path.poses)):
            prev_pose = self.estimated_path.poses[i-1].pose.position
            curr_pose = self.estimated_path.poses[i].pose.position
            
            dx = curr_pose.x - prev_pose.x
            dy = curr_pose.y - prev_pose.y
            dz = curr_pose.z - prev_pose.z
            
            distance = np.sqrt(dx*dx + dy*dy + dz*dz)
            total_distance += distance
            
        return total_distance

    def update_visualization(self):
        """주기적 시각화 업데이트 (간소화)"""
        try:
            # 추정된 경로 퍼블리시
            if self.estimated_path.poses:
                self.path_publisher.publish(self.estimated_path)
                
            # 궤적 마커 퍼블리시
            self.publish_trajectory_marker()
            
            # 통계 마커 퍼블리시
            self.publish_statistics_markers()
            
        except Exception as e:
            self.get_logger().error(f"시각화 업데이트 중 오류: {e}")

    def clear_visualization(self):
        """시각화 초기화 (간소화)"""
        self.estimated_path.poses.clear()
        
        # 마커 삭제
        delete_marker = Marker()
        delete_marker.action = Marker.DELETEALL
        self.pose_marker_publisher.publish(delete_marker)
        self.trajectory_publisher.publish(delete_marker)
        
        marker_array = MarkerArray()
        marker_array.markers.append(delete_marker)
        self.marker_publisher.publish(marker_array)
        
        self.get_logger().info("시각화 초기화 완료 (간소화된 방식)")


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
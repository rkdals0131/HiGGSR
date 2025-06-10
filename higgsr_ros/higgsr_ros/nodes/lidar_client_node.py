#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import numpy as np
import os
import time
from threading import Lock, Thread
import copy

from sensor_msgs.msg import PointCloud2
from geometry_msgs.msg import TransformStamped, PoseStamped
from nav_msgs.msg import Odometry
from tf2_ros import TransformBroadcaster
import tf_transformations
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from higgsr_interface.srv import RegisterScan
from higgsr_interface.msg import PointCloudInfo
from higgsr_ros.utils import ros_utils


class LidarClientNode(Node):
    """
    HiGGSR 라이다 클라이언트 노드
    - 라이다 토픽 수신 및 스캔 캡처
    - 캡처된 데이터를 이용한 스캔 정합 요청
    """
    
    def __init__(self):
        super().__init__('higgsr_lidar_client')
        self.get_logger().info("HiGGSR 라이다 클라이언트 노드 시작")

        # 파라미터 선언
        self.declare_parameter('lidar_topic', '/ouster/points')
        self.declare_parameter('publish_tf', True)
        self.declare_parameter('base_frame_id', 'base_link')
        self.declare_parameter('map_frame_id', 'map')
        self.declare_parameter('odom_frame_id', 'odom')
        
        # 내부 상태 변수
        self.latest_scan = None
        self.captured_scan = None
        self.processing_in_progress = False
        self.registration_lock = Lock()
        self.running = True

        # TF 브로드캐스터
        if self.get_parameter('publish_tf').get_parameter_value().bool_value:
            self.tf_broadcaster = TransformBroadcaster(self)

        # 라이다 토픽 구독
        lidar_topic = self.get_parameter('lidar_topic').get_parameter_value().string_value
        
        # QoS 프로파일 설정 (Best Effort)
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1  # Best Effort는 보통 depth 1을 사용
        )
        
        self.lidar_subscription = self.create_subscription(
            PointCloud2,
            lidar_topic,
            self.lidar_callback,
            qos_profile) # QoS 프로파일 적용
        self.get_logger().info(f"라이다 토픽 구독: {lidar_topic} (QoS: Best Effort)")

        # 서비스 클라이언트 생성
        self.register_scan_client = self.create_client(RegisterScan, 'register_scan')
        
        # 서비스 대기
        while not self.register_scan_client.wait_for_service(timeout_sec=5.0):
            self.get_logger().info('스캔 등록 서비스 대기 중...')
        
        self.get_logger().info("스캔 등록 서비스 연결됨")

        # 결과 퍼블리셔 생성
        self.pose_publisher = self.create_publisher(PoseStamped, 'higgsr_pose', 10)
        self.transform_publisher = self.create_publisher(TransformStamped, 'higgsr_transform', 10)

        # 키 입력 처리 스레드 시작
        self.input_thread = Thread(target=self.keyboard_input_handler, daemon=True)
        self.input_thread.start()

        self.get_logger().info("라이다 클라이언트 노드 초기화 완료")
        self.get_logger().info("엔터키를 눌러 스캔 캡처 및 정합을 실행하세요")

    def lidar_callback(self, msg: PointCloud2):
        """라이다 데이터 수신 콜백"""
        try:
            if not self.processing_in_progress:
                with self.registration_lock:
                    self.latest_scan = msg
                
        except Exception as e:
            self.get_logger().error(f"라이다 콜백 처리 중 오류: {e}")

    def keyboard_input_handler(self):
        """키보드 입력 처리 스레드"""
        while self.running:
            try:
                key_input = input()
                if key_input.strip() == '' or key_input.strip().lower() in ['space', ' ', 's']:
                    
                    if self.processing_in_progress:
                        self.get_logger().warn("이미 처리 중입니다. 처리 완료를 기다려주세요")
                        continue
                    
                    # 스캔 캡처
                    with self.registration_lock:
                        if self.latest_scan is not None:
                            self.captured_scan = copy.deepcopy(self.latest_scan)
                            self.processing_in_progress = True
                            
                            point_count = self.captured_scan.width * self.captured_scan.height
                            self.get_logger().info(f"스캔 캡처됨: {point_count} 포인트, 프레임 {self.captured_scan.header.frame_id}")
                        else:
                            self.get_logger().warn("사용 가능한 스캔이 없습니다")
                            continue
                    
                    # 캡처된 스캔으로 처리 시작
                    self.process_captured_scan()
                    
                elif key_input.strip().lower() in ['q', 'quit', 'exit']:
                    self.get_logger().info("종료 요청 받음")
                    self.running = False
                    break
                else:
                    self.get_logger().info("사용법: 엔터키(스캔 정합), 'q'(종료)")
            except (EOFError, KeyboardInterrupt):
                self.running = False
                break

    def process_captured_scan(self):
        """캡처된 스캔 처리"""
        try:
            if self.captured_scan is None:
                self.get_logger().error("캡처된 스캔이 없습니다")
                self.processing_in_progress = False
                return
            
            # 서비스 요청 생성
            request = RegisterScan.Request()
            request.live_scan_info = PointCloudInfo()
            request.live_scan_info.point_cloud = self.captured_scan
            request.live_scan_info.frame_id = self.captured_scan.header.frame_id
            request.live_scan_info.stamp = self.captured_scan.header.stamp

            self.get_logger().info("스캔 정합 처리 중...")
            
            # 서비스 호출
            future = self.register_scan_client.call_async(request)
            rclpy.spin_until_future_complete(self, future, timeout_sec=300.0)
            
            if future.result() is not None:
                response = future.result()
                
                if response.success:
                    self.process_registration_result(response)
                else:
                    self.get_logger().error(f"스캔 등록 실패: {response.message}")
            else:
                self.get_logger().error("서비스 호출 타임아웃")
                
        except Exception as e:
            self.get_logger().error(f"스캔 처리 중 예외: {e}")
        finally:
            self.processing_in_progress = False
            self.captured_scan = None
            self.get_logger().info("처리 완료. 다음 스캔을 위해 엔터키를 누르세요")

    def process_registration_result(self, response):
        """등록 결과 처리"""
        try:
            transform = response.estimated_transform
            score = response.score
            
            self.get_logger().info(f"스캔 등록 성공 - 점수: {score:.4f}")
            self.get_logger().info(f"변환: x={transform.transform.translation.x:.3f}, "
                                 f"y={transform.transform.translation.y:.3f}, "
                                 f"theta={np.rad2deg(2 * np.arcsin(transform.transform.rotation.z)):.2f}°")
            
            # Transform 및 Pose 퍼블리시
            self.transform_publisher.publish(transform)
            
            pose_msg = PoseStamped()
            pose_msg.header = transform.header
            pose_msg.pose.position.x = transform.transform.translation.x
            pose_msg.pose.position.y = transform.transform.translation.y
            pose_msg.pose.position.z = transform.transform.translation.z
            pose_msg.pose.orientation = transform.transform.rotation
            self.pose_publisher.publish(pose_msg)
            
            # TF 브로드캐스트
            if self.get_parameter('publish_tf').get_parameter_value().bool_value:
                self.tf_broadcaster.sendTransform(transform)
            
        except Exception as e:
            self.get_logger().error(f"결과 처리 중 오류: {e}")

    def destroy_node(self):
        """노드 종료 시 정리"""
        self.running = False
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    
    node = LidarClientNode()
    
    try:
        node.get_logger().info("라이다 클라이언트 실행 중...")
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('키보드 인터럽트 수신, 종료 중...')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main() 
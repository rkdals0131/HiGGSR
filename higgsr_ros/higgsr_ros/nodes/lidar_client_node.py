#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import numpy as np
from threading import Lock, Thread
import threading

from sensor_msgs.msg import PointCloud2
from geometry_msgs.msg import TransformStamped, PoseStamped
from nav_msgs.msg import Odometry
from tf2_ros import TransformBroadcaster
import tf_transformations

from higgsr_interface.srv import RegisterScan
from higgsr_interface.msg import PointCloudInfo
from higgsr_ros.utils import ros_utils


class LidarClientNode(Node):
    def __init__(self):
        super().__init__('higgsr_lidar_client')
        self.get_logger().info("HiGGSR 라이다 클라이언트 노드 시작")

        # 파라미터 선언
        self.declare_parameter('lidar_topic', '/ouster/points')
        self.declare_parameter('publish_tf', True)
        self.declare_parameter('base_frame_id', 'base_link')
        self.declare_parameter('map_frame_id', 'map_higgsr')
        self.declare_parameter('odom_frame_id', 'odom')
        
        # 내부 상태 변수
        self.latest_scan = None
        self.registration_lock = Lock()
        self.running = True

        # TF 브로드캐스터
        if self.get_parameter('publish_tf').get_parameter_value().bool_value:
            self.tf_broadcaster = TransformBroadcaster(self)

        # 라이다 토픽 구독
        lidar_topic = self.get_parameter('lidar_topic').get_parameter_value().string_value
        self.lidar_subscription = self.create_subscription(
            PointCloud2,
            lidar_topic,
            self.lidar_callback,
            10)
        self.get_logger().info(f"라이다 토픽 구독: {lidar_topic}")

        # 서비스 클라이언트 생성
        self.register_scan_client = self.create_client(RegisterScan, 'register_scan')

        # 결과 퍼블리셔 생성
        self.pose_publisher = self.create_publisher(PoseStamped, 'higgsr_pose', 10)
        self.transform_publisher = self.create_publisher(TransformStamped, 'higgsr_transform', 10)

        # 키 입력 처리 스레드 시작
        self.input_thread = Thread(target=self.keyboard_input_handler, daemon=True)
        self.input_thread.start()

        self.get_logger().info("라이다 클라이언트 노드 초기화 완료")
        self.get_logger().info("스페이스바나 엔터키를 눌러 스캔 정합을 요청하세요")

    def lidar_callback(self, msg: PointCloud2):
        """라이다 데이터 수신 콜백 - 최신 스캔 데이터만 저장"""
        try:
            with self.registration_lock:
                self.latest_scan = msg
                
        except Exception as e:
            self.get_logger().error(f"라이다 콜백 처리 중 오류: {e}")

    def keyboard_input_handler(self):
        """키보드 입력 처리 스레드"""
        while self.running:
            try:
                key_input = input()  # 엔터키 대기
                if key_input.strip() == '' or key_input.strip().lower() in ['space', ' ', 's']:
                    # 현재 스캔 상태 로그
                    with self.registration_lock:
                        if self.latest_scan is not None:
                            self.get_logger().info(f"현재 스캔 정합 요청 - 프레임 ID: {self.latest_scan.header.frame_id}, 시간: {self.latest_scan.header.stamp.sec}.{self.latest_scan.header.stamp.nanosec}")
                        else:
                            self.get_logger().warn("현재 사용 가능한 스캔이 없습니다")
                    self.trigger_registration()
                elif key_input.strip().lower() in ['q', 'quit', 'exit']:
                    self.get_logger().info("종료 요청 받음")
                    self.running = False
                    break
            except (EOFError, KeyboardInterrupt):
                self.running = False
                break

    def trigger_registration(self):
        """수동으로 정합 요청 트리거"""
        with self.registration_lock:
            if self.latest_scan is None:
                self.get_logger().warn("사용 가능한 라이다 데이터가 없습니다")
                return
            
            # 정합 요청할 스캔 데이터를 복사하여 고정
            scan_to_process = self.latest_scan
            self.get_logger().info(f"스캔 정합 요청 중... (포인트: {scan_to_process.width * scan_to_process.height})")
            self.call_registration_service_async(scan_to_process)

    def call_registration_service_async(self, msg: PointCloud2):
        """등록 서비스 비동기 호출"""
        if not self.register_scan_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn("RegisterScan 서비스를 사용할 수 없습니다")
            return

        request = RegisterScan.Request()
        request.live_scan_info = PointCloudInfo()
        request.live_scan_info.point_cloud = msg
        request.live_scan_info.frame_id = msg.header.frame_id
        request.live_scan_info.stamp = msg.header.stamp

        future = self.register_scan_client.call_async(request)
        future.add_done_callback(self.registration_service_callback)

    def registration_service_callback(self, future):
        """등록 서비스 응답 콜백"""
        try:
            response = future.result()
            if response.success:
                self.process_registration_result(response)
            else:
                self.get_logger().warn(f"스캔 등록 실패: {response.message}")
        except Exception as e:
            self.get_logger().error(f"등록 서비스 호출 오류: {e}")

    def process_registration_result(self, response):
        """등록 결과 처리 및 퍼블리시"""
        transform = response.estimated_transform
        score = response.score
        
        self.get_logger().info(f"스캔 등록 성공! 점수: {score:.3f}")
        self.get_logger().info(f"변환: tx={transform.transform.translation.x:.3f}, ty={transform.transform.translation.y:.3f}")
        
        # Transform 퍼블리시
        self.transform_publisher.publish(transform)
        
        # Pose 메시지 생성 및 퍼블리시
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

    def destroy_node(self):
        """노드 종료 시 정리"""
        self.running = False
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    
    node = LidarClientNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('키보드 인터럽트 수신, 종료 중...')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main() 
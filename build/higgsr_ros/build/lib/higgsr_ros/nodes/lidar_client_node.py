#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import numpy as np
import os
import time
from threading import Lock, Thread

from sensor_msgs.msg import PointCloud2
from geometry_msgs.msg import TransformStamped, PoseStamped
from nav_msgs.msg import Odometry
from tf2_ros import TransformBroadcaster
import tf_transformations

from higgsr_interface.srv import RegisterScan
from higgsr_interface.msg import PointCloudInfo
from higgsr_ros.utils import ros_utils


class LidarClientNode(Node):
    """
    HiGGSR 라이다 클라이언트 노드 (test_rviz_visualization.py 로직 기반)
    - 라이다 토픽 수신 및 저장
    - 키 입력으로 처리 트리거 (test_rviz_visualization.py와 동일한 방식)
    - 완벽하게 작동하는 test_rviz의 트리거 로직 사용
    """
    
    def __init__(self):
        super().__init__('higgsr_lidar_client')
        self.get_logger().info("HiGGSR 라이다 클라이언트 노드 시작 (test_rviz 로직 기반)")

        # 파라미터 선언
        self.declare_parameter('lidar_topic', '/ouster/points')
        self.declare_parameter('publish_tf', True)
        self.declare_parameter('base_frame_id', 'base_link')
        self.declare_parameter('map_frame_id', 'map')
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

        # 서비스 클라이언트 생성 (test_rviz_visualization.py와 동일)
        self.register_scan_client = self.create_client(RegisterScan, 'register_scan')
        
        # 서비스 대기 (test_rviz_visualization.py와 동일한 방식)
        while not self.register_scan_client.wait_for_service(timeout_sec=5.0):
            self.get_logger().info('스캔 등록 서비스 대기 중...')
        
        self.get_logger().info("스캔 등록 서비스 연결됨")

        # 결과 퍼블리셔 생성
        self.pose_publisher = self.create_publisher(PoseStamped, 'higgsr_pose', 10)
        self.transform_publisher = self.create_publisher(TransformStamped, 'higgsr_transform', 10)

        # 키 입력 처리 스레드 시작 (test_rviz_visualization.py와 동일한 방식)
        self.input_thread = Thread(target=self.keyboard_input_handler, daemon=True)
        self.input_thread.start()

        self.get_logger().info("라이다 클라이언트 노드 초기화 완료 (test_rviz 방식)")
        self.get_logger().info("🎯 스페이스바나 엔터키를 눌러 스캔 정합을 요청하세요!")
        self.get_logger().info("   - 완벽하게 작동하는 test_rviz_visualization.py와 동일한 트리거 방식 사용")

    def lidar_callback(self, msg: PointCloud2):
        """라이다 데이터 수신 콜백 - 최신 스캔 데이터만 저장"""
        try:
            with self.registration_lock:
                self.latest_scan = msg
                # 라이다 데이터 수신 상태를 간단히 로깅
                if self.latest_scan is not None:
                    point_count = self.latest_scan.width * self.latest_scan.height
                    # 너무 자주 로깅하지 않도록 조건부 로깅
                    if point_count > 0:
                        # 간단한 수신 확인만 (verbose하지 않게)
                        pass
                
        except Exception as e:
            self.get_logger().error(f"라이다 콜백 처리 중 오류: {e}")

    def keyboard_input_handler(self):
        """키보드 입력 처리 스레드 (test_rviz_visualization.py와 동일한 방식)"""
        while self.running:
            try:
                # test_rviz_visualization.py와 동일한 입력 대기 방식
                key_input = input()  # 엔터키 대기
                if key_input.strip() == '' or key_input.strip().lower() in ['space', ' ', 's']:
                    # 현재 스캔 상태 로그 (test_rviz와 동일한 방식)
                    with self.registration_lock:
                        if self.latest_scan is not None:
                            point_count = self.latest_scan.width * self.latest_scan.height
                            self.get_logger().info(f"🚀 스캔 정합 요청 시작!")
                            self.get_logger().info(f"   - 프레임 ID: {self.latest_scan.header.frame_id}")
                            self.get_logger().info(f"   - 포인트 수: {point_count}")
                            self.get_logger().info(f"   - 타임스탬프: {self.latest_scan.header.stamp.sec}.{self.latest_scan.header.stamp.nanosec}")
                        else:
                            self.get_logger().warn("❌ 현재 사용 가능한 스캔이 없습니다")
                            continue
                    
                    # test_rviz_visualization.py와 동일한 방식으로 처리 트리거
                    self.trigger_registration_test_rviz_style()
                    
                elif key_input.strip().lower() in ['q', 'quit', 'exit']:
                    self.get_logger().info("종료 요청 받음")
                    self.running = False
                    break
                else:
                    self.get_logger().info("💡 사용법:")
                    self.get_logger().info("   - 엔터키 또는 's' 입력: 스캔 정합 실행")
                    self.get_logger().info("   - 'q' 입력: 종료")
            except (EOFError, KeyboardInterrupt):
                self.running = False
                break

    def trigger_registration_test_rviz_style(self):
        """test_rviz_visualization.py와 동일한 방식으로 정합 요청 트리거"""
        with self.registration_lock:
            if self.latest_scan is None:
                self.get_logger().warn("사용 가능한 라이다 데이터가 없습니다")
                return
            
            # 정합 요청할 스캔 데이터를 복사하여 고정 (test_rviz와 동일한 방식)
            scan_to_process = self.latest_scan
            
            # test_rviz_visualization.py와 동일한 방식으로 서비스 요청 생성 및 호출
            self.call_registration_service_test_rviz_style(scan_to_process)

    def call_registration_service_test_rviz_style(self, msg: PointCloud2):
        """test_rviz_visualization.py와 동일한 방식으로 등록 서비스 호출"""
        try:
            # test_rviz_visualization.py와 동일한 방식으로 서비스 요청 생성
            request = RegisterScan.Request()
            request.live_scan_info = PointCloudInfo()
            request.live_scan_info.point_cloud = msg
            request.live_scan_info.frame_id = msg.header.frame_id
            request.live_scan_info.stamp = msg.header.stamp

            self.get_logger().info("🔄 처리 요청 전송 중... (시간이 걸릴 수 있습니다)")
            self.get_logger().info("   - test_rviz_visualization.py와 동일한 방식 사용")
            
            # test_rviz_visualization.py와 동일한 방식으로 서비스 호출
            future = self.register_scan_client.call_async(request)
            
            # 결과 대기 (test_rviz_visualization.py와 동일한 방식)
            rclpy.spin_until_future_complete(self, future, timeout_sec=300.0)  # 5분 타임아웃
            
            if future.result() is not None:
                response = future.result()
                
                if response.success:
                    self.process_registration_result_test_rviz_style(response)
                else:
                    self.get_logger().error(f"❌ 스캔 등록 실패: {response.message}")
            else:
                self.get_logger().error("❌ 서비스 호출 타임아웃 또는 실패")
                
        except Exception as e:
            self.get_logger().error(f"❌ 등록 서비스 호출 중 예외: {e}")

    def process_registration_result_test_rviz_style(self, response):
        """test_rviz_visualization.py와 동일한 방식으로 등록 결과 처리"""
        try:
            transform = response.estimated_transform
            score = response.score
            
            # test_rviz_visualization.py와 동일한 방식으로 결과 출력
            self.get_logger().info("✅ 스캔 등록 성공! (test_rviz 방식)")
            self.get_logger().info(f"   점수: {score:.4f}")
            
            # 변환 결과 출력 (test_rviz_visualization.py와 동일한 방식)
            self.get_logger().info(f"   변환 결과:")
            self.get_logger().info(f"     위치: x={transform.transform.translation.x:.3f}, y={transform.transform.translation.y:.3f}, z={transform.transform.translation.z:.3f}")
            self.get_logger().info(f"     회전: x={transform.transform.rotation.x:.3f}, y={transform.transform.rotation.y:.3f}, z={transform.transform.rotation.z:.3f}, w={transform.transform.rotation.w:.3f}")
            
            # test_rviz_visualization.py와 동일한 방식으로 시각화 안내
            self.get_logger().info("🎯 RViz2에서 시각화 결과를 확인하세요!")
            self.get_logger().info("   - 글로벌 맵: 흰색 포인트클라우드")
            self.get_logger().info("   - 변환된 스캔: 빨간색 포인트클라우드")
            self.get_logger().info("   - 글로벌 키포인트: 파란색 구")
            self.get_logger().info("   - 변환된 스캔 키포인트: 녹색 구")
            
            # Transform 퍼블리시 (기존 로직 유지)
            self.transform_publisher.publish(transform)
            
            # Pose 메시지 생성 및 퍼블리시 (기존 로직 유지)
            pose_msg = PoseStamped()
            pose_msg.header = transform.header
            pose_msg.pose.position.x = transform.transform.translation.x
            pose_msg.pose.position.y = transform.transform.translation.y
            pose_msg.pose.position.z = transform.transform.translation.z
            pose_msg.pose.orientation = transform.transform.rotation
            self.pose_publisher.publish(pose_msg)
            
            # TF 브로드캐스트 (기존 로직 유지)
            if self.get_parameter('publish_tf').get_parameter_value().bool_value:
                self.tf_broadcaster.sendTransform(transform)
                
            self.get_logger().info("💡 다음 스캔 정합을 위해 엔터키를 다시 누르세요!")
            
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
        # test_rviz_visualization.py와 동일한 방식으로 실행
        node.get_logger().info("🚀 라이다 클라이언트 실행 중... (test_rviz 방식)")
        node.get_logger().info("   엔터키를 눌러 스캔 정합을 실행하세요!")
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('키보드 인터럽트 수신, 종료 중...')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main() 
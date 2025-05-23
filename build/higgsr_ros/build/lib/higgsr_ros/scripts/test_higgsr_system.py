#!/usr/bin/env python3
"""
HiGGSR ROS2 시스템 테스트 스크립트

이 스크립트는 HiGGSR 시스템의 기본 동작을 테스트합니다:
1. 서비스 가용성 확인
2. 토픽 발행 상태 확인
3. 기본 기능 테스트
"""

import rclpy
from rclpy.node import Node
import time
import numpy as np

from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header
from higgsr_interface.srv import SetGlobalMap, RegisterScan
from higgsr_interface.msg import PointCloudInfo


class HiGGSRSystemTester(Node):
    def __init__(self):
        super().__init__('higgsr_system_tester')
        self.get_logger().info("HiGGSR 시스템 테스트 시작")

        # 서비스 클라이언트 생성
        self.set_global_map_client = self.create_client(SetGlobalMap, 'set_global_map')
        self.register_scan_client = self.create_client(RegisterScan, 'register_scan')

    def wait_for_services(self, timeout_sec=10.0):
        """서비스 가용성 확인"""
        self.get_logger().info("서비스 가용성 확인 중...")
        
        if not self.set_global_map_client.wait_for_service(timeout_sec=timeout_sec):
            self.get_logger().error("SetGlobalMap 서비스를 찾을 수 없습니다")
            return False
            
        if not self.register_scan_client.wait_for_service(timeout_sec=timeout_sec):
            self.get_logger().error("RegisterScan 서비스를 찾을 수 없습니다")
            return False
            
        self.get_logger().info("모든 서비스가 사용 가능합니다")
        return True

    def create_test_point_cloud(self, num_points=1000, size=10.0):
        """테스트용 포인트 클라우드 생성"""
        # 랜덤 포인트 생성
        points = np.random.uniform(-size/2, size/2, (num_points, 3)).astype(np.float32)
        
        # PointCloud2 메시지 생성
        msg = PointCloud2()
        msg.header = Header()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "base_link"
        
        msg.height = 1
        msg.width = num_points
        msg.is_dense = True
        msg.is_bigendian = False
        
        # 필드 정의
        fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
        ]
        msg.fields = fields
        msg.point_step = 12  # 3 floats * 4 bytes
        msg.row_step = msg.point_step * num_points
        
        # 데이터 설정
        msg.data = points.tobytes()
        
        return msg

    def test_set_global_map(self):
        """글로벌 맵 설정 테스트"""
        self.get_logger().info("글로벌 맵 설정 테스트 시작...")
        
        # 테스트 포인트 클라우드 생성
        point_cloud = self.create_test_point_cloud(1500, 15.0)
        
        # 요청 생성
        request = SetGlobalMap.Request()
        request.global_map_info = PointCloudInfo()
        request.global_map_info.point_cloud = point_cloud
        request.global_map_info.frame_id = "base_link"
        request.global_map_info.stamp = self.get_clock().now().to_msg()
        
        # 서비스 호출
        future = self.set_global_map_client.call_async(request)
        
        # 결과 대기
        rclpy.spin_until_future_complete(self, future, timeout_sec=30.0)
        
        if future.done():
            response = future.result()
            if response.success:
                self.get_logger().info("✅ 글로벌 맵 설정 성공")
                return True
            else:
                self.get_logger().error(f"❌ 글로벌 맵 설정 실패: {response.message}")
                return False
        else:
            self.get_logger().error("❌ 글로벌 맵 설정 서비스 호출 타임아웃")
            return False

    def test_register_scan(self):
        """스캔 등록 테스트"""
        self.get_logger().info("스캔 등록 테스트 시작...")
        
        # 테스트 라이브 스캔 생성 (글로벌 맵과 유사하지만 약간 다른 위치)
        point_cloud = self.create_test_point_cloud(800, 12.0)
        
        # 요청 생성
        request = RegisterScan.Request()
        request.live_scan_info = PointCloudInfo()
        request.live_scan_info.point_cloud = point_cloud
        request.live_scan_info.frame_id = "base_link"
        request.live_scan_info.stamp = self.get_clock().now().to_msg()
        
        # 서비스 호출
        future = self.register_scan_client.call_async(request)
        
        # 결과 대기
        rclpy.spin_until_future_complete(self, future, timeout_sec=60.0)
        
        if future.done():
            response = future.result()
            if response.success:
                transform = response.estimated_transform
                self.get_logger().info("✅ 스캔 등록 성공")
                self.get_logger().info(f"   점수: {response.score:.3f}")
                self.get_logger().info(f"   변환: x={transform.transform.translation.x:.3f}, "
                                      f"y={transform.transform.translation.y:.3f}, "
                                      f"z={transform.transform.translation.z:.3f}")
                return True
            else:
                self.get_logger().error(f"❌ 스캔 등록 실패: {response.message}")
                return False
        else:
            self.get_logger().error("❌ 스캔 등록 서비스 호출 타임아웃")
            return False

    def run_tests(self):
        """전체 테스트 실행"""
        self.get_logger().info("=" * 50)
        self.get_logger().info("HiGGSR 시스템 테스트 시작")
        self.get_logger().info("=" * 50)
        
        # 1. 서비스 가용성 확인
        if not self.wait_for_services():
            self.get_logger().error("테스트 중단: 서비스를 찾을 수 없습니다")
            return False
        
        # 2. 글로벌 맵 설정 테스트
        if not self.test_set_global_map():
            self.get_logger().error("테스트 중단: 글로벌 맵 설정 실패")
            return False
        
        # 잠시 대기 (시스템이 글로벌 맵을 처리할 시간)
        time.sleep(2.0)
        
        # 3. 스캔 등록 테스트
        if not self.test_register_scan():
            self.get_logger().error("테스트 중단: 스캔 등록 실패")
            return False
        
        self.get_logger().info("=" * 50)
        self.get_logger().info("🎉 모든 테스트 통과!")
        self.get_logger().info("=" * 50)
        return True


def main(args=None):
    rclpy.init(args=args)
    
    tester = HiGGSRSystemTester()
    
    try:
        success = tester.run_tests()
        if success:
            print("\n✅ HiGGSR 시스템이 정상적으로 작동합니다!")
        else:
            print("\n❌ HiGGSR 시스템에 문제가 있습니다.")
            return 1
            
    except KeyboardInterrupt:
        tester.get_logger().info("테스트가 사용자에 의해 중단되었습니다")
    except Exception as e:
        tester.get_logger().error(f"테스트 중 예외 발생: {e}")
        return 1
    finally:
        tester.destroy_node()
        rclpy.shutdown()
    
    return 0


if __name__ == '__main__':
    exit(main()) 
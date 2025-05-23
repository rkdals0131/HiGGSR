#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import os
import time

from higgsr_interface.srv import ProcessFiles


class RVizVisualizationTester(Node):
    def __init__(self):
        super().__init__('rviz_visualization_tester')
        self.get_logger().info("RViz2 시각화 테스터 노드 시작")
        
        # 클라이언트 생성
        self.client = self.create_client(ProcessFiles, 'process_files')
        
        # 서비스 대기
        while not self.client.wait_for_service(timeout_sec=5.0):
            self.get_logger().info('파일 처리 서비스 대기 중...')
        
        self.get_logger().info("파일 처리 서비스 연결됨")

    def test_visualization(self, global_map_path=None, live_scan_path=None):
        """시각화 테스트 실행"""
        
        # 기본 파일 경로 설정
        if global_map_path is None or live_scan_path is None:
            # 기본 데이터 파일 경로
            package_share = "/home/user1/ROS2_Workspace/higgsros_ws/src/higgsr_ros/Data"
            default_global_map = os.path.join(package_share, "around_singong - Cloud.ply")
            default_live_scan = os.path.join(package_share, "around_singong_ply/001355.ply")
            
            global_map_path = global_map_path or default_global_map
            live_scan_path = live_scan_path or default_live_scan
        
        # 파일 존재 확인
        if not os.path.exists(global_map_path):
            self.get_logger().error(f"글로벌 맵 파일을 찾을 수 없음: {global_map_path}")
            return False
        
        if not os.path.exists(live_scan_path):
            self.get_logger().error(f"라이브 스캔 파일을 찾을 수 없음: {live_scan_path}")
            return False
        
        # 서비스 요청 생성
        request = ProcessFiles.Request()
        request.global_map_filepath = global_map_path
        request.live_scan_filepath = live_scan_path
        request.algorithm_config_json = ""  # 기본 설정 사용
        
        self.get_logger().info(f"처리 요청 전송:")
        self.get_logger().info(f"  글로벌 맵: {global_map_path}")
        self.get_logger().info(f"  라이브 스캔: {live_scan_path}")
        
        # 서비스 호출
        try:
            future = self.client.call_async(request)
            self.get_logger().info("파일 처리 중... (시간이 걸릴 수 있습니다)")
            
            # 결과 대기
            rclpy.spin_until_future_complete(self, future)
            
            if future.result() is not None:
                response = future.result()
                
                if response.success:
                    self.get_logger().info("✅ 파일 처리 성공!")
                    self.get_logger().info(f"  점수: {response.score:.4f}")
                    self.get_logger().info(f"  처리 시간: {response.processing_time_seconds:.2f}초")
                    self.get_logger().info(f"  계산 반복 수: {response.total_calc_iterations}")
                    
                    # 변환 결과 출력
                    tf = response.estimated_transform
                    self.get_logger().info(f"  변환 결과:")
                    self.get_logger().info(f"    위치: x={tf.transform.translation.x:.3f}, y={tf.transform.translation.y:.3f}, z={tf.transform.translation.z:.3f}")
                    self.get_logger().info(f"    회전: x={tf.transform.rotation.x:.3f}, y={tf.transform.rotation.y:.3f}, z={tf.transform.rotation.z:.3f}, w={tf.transform.rotation.w:.3f}")
                    
                    self.get_logger().info("🎯 RViz2에서 시각화 결과를 확인하세요!")
                    self.get_logger().info("   - 글로벌 맵: 흰색 포인트클라우드")
                    self.get_logger().info("   - 변환된 스캔: 빨간색 포인트클라우드")
                    self.get_logger().info("   - 글로벌 키포인트: 파란색 구")
                    self.get_logger().info("   - 변환된 스캔 키포인트: 녹색 구")
                    
                    return True
                else:
                    self.get_logger().error(f"❌ 파일 처리 실패: {response.message}")
                    return False
            else:
                self.get_logger().error("❌ 서비스 호출 실패")
                return False
                
        except Exception as e:
            self.get_logger().error(f"❌ 예외 발생: {e}")
            return False

def main(args=None):
    rclpy.init(args=args)
    
    node = RVizVisualizationTester()
    
    try:
        # 시각화 테스트 실행
        success = node.test_visualization()
        
        if success:
            node.get_logger().info("테스트 완료. RViz2에서 결과를 확인하세요.")
            # 노드를 유지해서 시각화를 계속 볼 수 있도록
            node.get_logger().info("Ctrl+C를 눌러 종료하세요.")
            rclpy.spin(node)
        else:
            node.get_logger().error("테스트 실패")
            
    except KeyboardInterrupt:
        node.get_logger().info('키보드 인터럽트 수신, 종료 중...')
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main() 
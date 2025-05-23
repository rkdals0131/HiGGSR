#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import json
import argparse
import sys
import os

from higgsr_interface.srv import ProcessFiles


class FileProcessorClientNode(Node):
    def __init__(self):
        super().__init__('file_processor_client_node')
        self.get_logger().info("파일 처리 클라이언트 노드 시작.")
        
        # ProcessFiles 서비스 클라이언트 생성
        self.client = self.create_client(ProcessFiles, 'process_files')
        
        # 서비스 대기
        while not self.client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('ProcessFiles 서비스를 기다리는 중...')
        self.get_logger().info("ProcessFiles 서비스 연결됨.")

    def call_process_files(self, global_map_filepath, live_scan_filepath, algorithm_config_json=""):
        """파일 처리 서비스를 호출합니다."""
        
        # 요청 생성
        request = ProcessFiles.Request()
        request.global_map_filepath = global_map_filepath
        request.live_scan_filepath = live_scan_filepath
        request.algorithm_config_json = algorithm_config_json
        
        self.get_logger().info(f"파일 처리 요청 전송:")
        self.get_logger().info(f"  전역 맵: {global_map_filepath}")
        self.get_logger().info(f"  라이브 스캔: {live_scan_filepath}")
        if algorithm_config_json.strip():
            self.get_logger().info(f"  알고리즘 설정: 제공됨 (JSON)")
        else:
            self.get_logger().info(f"  알고리즘 설정: 노드 기본값 사용")
        
        # 서비스 호출
        future = self.client.call_async(request)
        return future

    def print_result(self, response):
        """서비스 응답을 출력합니다."""
        if response.success:
            self.get_logger().info("=== 파일 처리 성공 ===")
            self.get_logger().info(f"점수: {response.score}")
            self.get_logger().info(f"처리 시간: {response.processing_time_seconds:.2f} 초")
            self.get_logger().info(f"계산 반복 수: {response.total_calc_iterations}")
            
            transform = response.estimated_transform
            self.get_logger().info("추정된 변환:")
            self.get_logger().info(f"  프레임: {transform.header.frame_id} -> {transform.child_frame_id}")
            self.get_logger().info(f"  Translation: [{transform.transform.translation.x:.3f}, "
                                 f"{transform.transform.translation.y:.3f}, "
                                 f"{transform.transform.translation.z:.3f}]")
            self.get_logger().info(f"  Rotation: [{transform.transform.rotation.x:.3f}, "
                                 f"{transform.transform.rotation.y:.3f}, "
                                 f"{transform.transform.rotation.z:.3f}, "
                                 f"{transform.transform.rotation.w:.3f}]")
        else:
            self.get_logger().error("=== 파일 처리 실패 ===")
            self.get_logger().error(f"오류 메시지: {response.message}")


def main(args=None):
    """메인 함수 - 명령줄 인터페이스 제공"""
    
    # 명령줄 인자 파싱
    if args is None:
        args = sys.argv[1:]
    
    parser = argparse.ArgumentParser(description='HiGGSR 파일 처리 클라이언트')
    parser.add_argument('--global-map', type=str, required=True, 
                       help='전역 맵 포인트 클라우드 파일 경로')
    parser.add_argument('--live-scan', type=str, required=True,
                       help='라이브 스캔 포인트 클라우드 파일 경로')
    parser.add_argument('--config-json', type=str, default="",
                       help='알고리즘 설정 JSON 문자열 (옵션)')
    parser.add_argument('--config-file', type=str, default="",
                       help='알고리즘 설정 JSON 파일 경로 (옵션)')
    
    # 샘플 설정들
    parser.add_argument('--quick', action='store_true',
                       help='빠른 처리를 위한 설정 사용')
    parser.add_argument('--accurate', action='store_true',
                       help='정확한 처리를 위한 설정 사용')
    
    try:
        parsed_args = parser.parse_args(args)
    except SystemExit:
        return
    
    # 파일 존재 여부 확인
    if not os.path.exists(parsed_args.global_map):
        print(f"오류: 전역 맵 파일을 찾을 수 없습니다: {parsed_args.global_map}")
        return
    
    if not os.path.exists(parsed_args.live_scan):
        print(f"오류: 라이브 스캔 파일을 찾을 수 없습니다: {parsed_args.live_scan}")
        return
    
    # 알고리즘 설정 준비
    algorithm_config_json = ""
    
    if parsed_args.config_file:
        try:
            with open(parsed_args.config_file, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
                algorithm_config_json = json.dumps(config_data)
        except Exception as e:
            print(f"설정 파일 로딩 오류: {e}")
            return
    elif parsed_args.config_json:
        algorithm_config_json = parsed_args.config_json
    elif parsed_args.quick:
        # 빠른 처리 설정
        quick_config = {
            "grid_size": 0.4,
            "keypoint_density_threshold": 0.15,
            "level_configs": [
                {
                    "grid_division": [4, 4],
                    "search_area_type": "full_map",
                    "theta_range_deg": [0, 359], "theta_search_steps": 24,
                    "correspondence_distance_threshold_factor": 10.0,
                    "tx_ty_search_steps_per_cell": [5, 5]
                }
            ],
            "visualization": {
                "enable_pillar_maps": False,
                "enable_2d_keypoints": False,
                "enable_super_grid_heatmap": False,
                "enable_3d_result": True
            }
        }
        algorithm_config_json = json.dumps(quick_config)
    elif parsed_args.accurate:
        # 정확한 처리 설정
        accurate_config = {
            "grid_size": 0.1,
            "keypoint_density_threshold": 0.05,
            "level_configs": [
                {
                    "grid_division": [8, 8],
                    "search_area_type": "full_map",
                    "theta_range_deg": [0, 359], "theta_search_steps": 72,
                    "correspondence_distance_threshold_factor": 5.0,
                    "tx_ty_search_steps_per_cell": [15, 15]
                },
                {
                    "grid_division": [10, 10],
                    "search_area_type": "relative_to_map", "area_ratio_or_size": 0.3,
                    "theta_range_deg_relative": [0, 359], "theta_search_steps": 72,
                    "correspondence_distance_threshold_factor": 3.0,
                    "tx_ty_search_steps_per_cell": [15, 15]
                }
            ],
            "visualization": {
                "enable_pillar_maps": True,
                "enable_2d_keypoints": True,
                "enable_super_grid_heatmap": True,
                "enable_3d_result": True
            }
        }
        algorithm_config_json = json.dumps(accurate_config)
    
    # ROS 초기화
    rclpy.init()
    
    try:
        # 클라이언트 노드 생성
        client_node = FileProcessorClientNode()
        
        # 서비스 호출
        future = client_node.call_process_files(
            parsed_args.global_map,
            parsed_args.live_scan,
            algorithm_config_json
        )
        
        # 응답 대기
        rclpy.spin_until_future_complete(client_node, future)
        
        if future.result() is not None:
            client_node.print_result(future.result())
        else:
            client_node.get_logger().error("서비스 호출 실패")
        
    except KeyboardInterrupt:
        print('키보드 인터럽트 수신, 종료 중...')
    finally:
        try:
            client_node.destroy_node()
        except:
            pass
        rclpy.try_shutdown()


if __name__ == '__main__':
    main() 
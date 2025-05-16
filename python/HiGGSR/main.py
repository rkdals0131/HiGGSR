import sys
import os

# 현재 main.py 파일의 절대 경로를 기준으로 src 디렉토리 경로를 계산
# main.py 위치: <workspace>/src/HiGGSR/python/HiGGSR/main.py
# src 디렉토리: <workspace>/src
_main_py_abs_path = os.path.abspath(__file__)
_higgsr_module_dir = os.path.dirname(_main_py_abs_path)  # .../src/HiGGSR/python/HiGGSR
_python_dir = os.path.dirname(_higgsr_module_dir)       # .../src/HiGGSR/python
_higgsr_pkg_root_in_src = os.path.dirname(_python_dir) # .../src/HiGGSR
_src_dir = os.path.dirname(_higgsr_pkg_root_in_src)    # .../src

if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

import numpy as np
import multiprocessing
import argparse
import time

from HiGGSR import core
from HiGGSR import visualization as viz

def main():
    """
    HiGGSR 메인 함수 - 명령줄 인터페이스 제공
    """
    parser = argparse.ArgumentParser(description='HiGGSR - 계층적 전역 그리드 탐색 및 정합')
    parser.add_argument('--global-map', type=str, help='전역 맵 포인트 클라우드 파일 경로')
    parser.add_argument('--live-scan', type=str, help='라이브 스캔 포인트 클라우드 파일 경로')
    parser.add_argument('--grid-size', type=float, default=0.2, help='그리드 셀 크기 (기본값: 0.2)')
    parser.add_argument('--min-points', type=int, default=3, help='밀도 계산에 필요한 최소 포인트 수 (기본값: 3)')
    parser.add_argument('--density-metric', type=str, default='std', choices=['std', 'range'], help='밀도 메트릭 (기본값: std)')
    parser.add_argument('--keypoint-threshold', type=float, default=0.1, help='키포인트 밀도 임계값 (기본값: 0.1)')
    parser.add_argument('--processes', type=int, default=None, help='병렬 처리에 사용할 프로세스 수 (기본값: 시스템 CPU 수)')
    parser.add_argument('--visualize', action='store_true', help='결과 시각화 활성화')
    parser.add_argument('--use-cpp', action=argparse.BooleanOptionalAction, default=False, help='C++ 확장 사용 여부 (기본값: 사용)')
    
    args = parser.parse_args()
    
    # HiGGSR 코어에 C++ 사용 여부 전달
    if hasattr(core, 'set_use_cpp_extensions'):
        core.set_use_cpp_extensions(args.use_cpp)
    else:
        print("WARNING: core.set_use_cpp_extensions not found. Defaulting to core module's behavior.")
    
    # 명령줄에서 파일 경로가 제공되면 사용, 그렇지 않으면 기본값 사용
    if args.global_map and args.live_scan:
        GLOBAL_MAP_FILEPATH = args.global_map
        LIVE_SCAN_FILEPATH = args.live_scan
    else:
        # 기본 데이터 파일 경로 설정
        BASE_DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "Data")
        GLOBAL_MAP_FILEPATH = os.path.join(BASE_DATA_PATH, "around_singong - Cloud.ply")
        LIVE_SCAN_FILEPATH = os.path.join(BASE_DATA_PATH, "around_singong_ply/001355.ply")
    
    # 파일 존재 여부 확인
    if not os.path.exists(GLOBAL_MAP_FILEPATH):
        exit(f"오류: 글로벌 맵 파일을 찾을 수 없습니다: {GLOBAL_MAP_FILEPATH}")
    if not os.path.exists(LIVE_SCAN_FILEPATH):
        exit(f"오류: 라이브 스캔 파일을 찾을 수 없습니다: {LIVE_SCAN_FILEPATH}")

    #=========================================================================
    # 알고리즘 파라미터 설정
    #=========================================================================
    
    # 그리드 및 밀도 계산 설정
    GRID_CELL_SIZE = args.grid_size                                           
    MIN_POINTS_FOR_DENSITY_CALC = args.min_points                                 
    DENSITY_METRIC = args.density_metric                                          
    KEYPOINT_DENSITY_THRESHOLD_STD = args.keypoint_threshold                            
    
    # 멀티프로세싱 설정
    NUM_PROCESSES = args.processes 
    
    # 시각화 설정 - 명령줄 인자로 받은 값을 사용할 경우 args.visualize 대신 True/False로 설정
    # 여기서 시각화 옵션을 직접 설정할 수 있습니다 (True/False)
    USE_COMMAND_LINE_VIZ = False  # 명령줄에서 --visualize 인자를 받을지 여부
    
    if USE_COMMAND_LINE_VIZ:
        VISUALIZE_PILLAR_MAPS = args.visualize
        VISUALIZE_2D_KEYPOINTS = args.visualize
        VISUALIZE_SUPER_GRID_HEATMAP = args.visualize 
        VISUALIZE_3D_RESULT = args.visualize
    else:
        # 각 시각화 옵션을 직접 설정 (True/False)
        VISUALIZE_PILLAR_MAPS = False
        VISUALIZE_2D_KEYPOINTS = True
        VISUALIZE_SUPER_GRID_HEATMAP = True 
        VISUALIZE_3D_RESULT = True

    # 계층적 탐색 설정
    LEVEL_CONFIGS = [
        { 
            "grid_division": (6, 6), 
            "search_area_type": "full_map", 
            "theta_range_deg": (0, 359), "theta_search_steps": 48, 
            "correspondence_distance_threshold_factor": 7.0,
            "tx_ty_search_steps_per_cell": (10, 10) 
        },
        { 
            "grid_division": (7, 7), 
            "search_area_type": "relative_to_map", "area_ratio_or_size": 0.4,
            "theta_range_deg_relative": (0, 359), "theta_search_steps": 48,
            "correspondence_distance_threshold_factor": 5.0,
            "tx_ty_search_steps_per_cell": (10, 10)
        },
        { 
            "grid_division": (4, 4), 
            "search_area_type": "absolute_size", "area_ratio_or_size": (40.0, 40.0),
            "theta_range_deg_relative": (0, 359), "theta_search_steps": 48,
            "correspondence_distance_threshold_factor": 2.5,
            "tx_ty_search_steps_per_cell": (10, 10)
        }
    ]
    NUM_CANDIDATES_PER_LEVEL = 3 
    MIN_CANDIDATE_SEPARATION_FACTOR = 1.5 
    
    #=========================================================================
    # 1. 데이터 로딩
    #=========================================================================
    print("1. 데이터 로딩 중...")
    
    # 실제 데이터 로딩
    global_map_points_3d = core.load_point_cloud_from_file(GLOBAL_MAP_FILEPATH)
    live_scan_points_3d = core.load_point_cloud_from_file(LIVE_SCAN_FILEPATH)
    
    # 데이터 유효성 검사
    if global_map_points_3d.shape[0]==0 or live_scan_points_3d.shape[0]==0: 
        exit("오류: 포인트 클라우드 로딩 실패.")

    #=========================================================================
    # 2. Pillar Map 생성
    #=========================================================================
    print("\n2. Pillar Map 생성 중...")
    # 글로벌 맵과 스캔 데이터에서 높이 분산 맵 생성
    density_map_global, x_edges_global, y_edges_global = core.create_2d_height_variance_map(
        global_map_points_3d, GRID_CELL_SIZE, MIN_POINTS_FOR_DENSITY_CALC, DENSITY_METRIC
    )
    density_map_scan, x_edges_scan, y_edges_scan = core.create_2d_height_variance_map(
        live_scan_points_3d, GRID_CELL_SIZE, MIN_POINTS_FOR_DENSITY_CALC, DENSITY_METRIC
    )
    
    # 맵 유효성 검사
    if density_map_global.size==0 or density_map_scan.size==0: 
        exit("오류: Pillar Map 생성 실패.")
    
    print(f"  Global Pillar Map: {density_map_global.shape}, Scan Pillar Map: {density_map_scan.shape}")
    
    # Pillar Map 시각화 (옵션)
    if VISUALIZE_PILLAR_MAPS:
        viz.visualize_density_map(density_map_global, x_edges_global, y_edges_global, title_suffix=f"Global Map ({DENSITY_METRIC})")
        viz.visualize_density_map(density_map_scan, x_edges_scan, y_edges_scan, title_suffix=f"Live Scan ({DENSITY_METRIC})")

    #=========================================================================
    # 3. 키포인트 추출
    #=========================================================================
    print("\n3. 키포인트 추출 중...")
    # 밀도 기반 키포인트 추출
    current_keypoint_threshold = KEYPOINT_DENSITY_THRESHOLD_STD 
    global_keypoints = core.extract_high_density_keypoints(density_map_global, x_edges_global, y_edges_global, current_keypoint_threshold)
    scan_keypoints = core.extract_high_density_keypoints(density_map_scan, x_edges_scan, y_edges_scan, current_keypoint_threshold)
    
    print(f"  Global Map Keypoints: {global_keypoints.shape[0]}, Live Scan Keypoints: {scan_keypoints.shape[0]}")
    
    # 키포인트 유효성 검사
    if global_keypoints.shape[0]<10 or scan_keypoints.shape[0]<5:
        print("경고: 추출된 키포인트 수가 매우 적습니다.")
        if global_keypoints.shape[0]==0 or scan_keypoints.shape[0]==0: 
            exit("오류: 키포인트가 없어 정합 불가.")

    #=========================================================================
    # 4. 계층적 적응형 전역 정합 수행
    #=========================================================================
    print("\n4. 계층적 적응형 전역 정합 수행 중...")
    start_time = time.time()
    final_transform_dict, final_score, all_levels_visualization_data, total_hierarchical_time, total_calc_iterations = core.hierarchical_adaptive_search(
        global_keypoints, scan_keypoints,
        x_edges_global, y_edges_global, 
        LEVEL_CONFIGS,
        NUM_CANDIDATES_PER_LEVEL, 
        MIN_CANDIDATE_SEPARATION_FACTOR,
        GRID_CELL_SIZE, 
        num_processes=NUM_PROCESSES
    )
    total_time = time.time() - start_time

    # 정합 결과 추출
    est_tx = final_transform_dict['tx']
    est_ty = final_transform_dict['ty']
    est_theta_deg = final_transform_dict['theta_deg']
    best_score = final_score 
    reg_time = total_hierarchical_time
    
    #=========================================================================
    # 5. 결과 출력 및 시각화
    #=========================================================================
    print("\n--- 정합 결과 ---")
    print(f"  추정된 변환: tx={est_tx:.3f}, ty={est_ty:.3f}, theta={est_theta_deg:.2f} deg")
    print(f"  최고 점수: {best_score}")
    print(f"  정합 소요 시간: {reg_time:.2f} 초")
    print(f"  총 계산된 변환 후보 수: {total_calc_iterations}")
    print(f"  전체 소요 시간: {total_time:.2f} 초")

    # 최종 변환 행렬 생성 및 출력
    if best_score > -1: 
        final_transform_matrix_4x4 = core.create_transform_matrix_4x4(est_tx, est_ty, est_theta_deg)
        print("\n -- 최종 정합 결과 -- (4x4 동차 변환 행렬):")
        print(final_transform_matrix_4x4)

    # 2D 키포인트 정합 결과 시각화
    if VISUALIZE_2D_KEYPOINTS and best_score > -1 :
        print("\n5.1. 2D 키포인트 정합 결과 시각화 중...")
        transformed_scan_keypoints_for_viz_np = core.apply_transform_to_keypoints_numba(
            scan_keypoints, est_tx, est_ty, np.deg2rad(est_theta_deg)
        )
        viz.visualize_2d_keypoint_registration(
            global_keypoints, scan_keypoints, transformed_scan_keypoints_for_viz_np, 
            x_edges_global, y_edges_global, title=f"2D Keypoint Registration (Score: {best_score})"
        )

    # 계층적 탐색 결과 히트맵 시각화
    if VISUALIZE_SUPER_GRID_HEATMAP and best_score > -1 and all_levels_visualization_data:
        print("\n5.2. 계층적 탐색 결과 히트맵 시각화 중...")
        for level_data in all_levels_visualization_data:
            print(f"  Visualizing Level {level_data['level']} heatmap...")
            next_config_index = level_data['level']
            next_lvl_cfg_to_pass = LEVEL_CONFIGS[next_config_index] if next_config_index < len(LEVEL_CONFIGS) else None

            viz.visualize_super_grid_scores(
                density_map_global, 
                x_edges_global,     
                y_edges_global,     
                level_data['all_raw_cell_infos_this_level'], 
                level_data['searched_areas_details'], 
                x_edges_global, 
                y_edges_global, 
                level_data['selected_candidates_after_nms'],
                next_lvl_cfg_to_pass,
                (x_edges_global, y_edges_global), 
                title_suffix=f"Level {level_data['level']}"
            )

    # 3D 포인트 클라우드 정합 결과 시각화
    if VISUALIZE_3D_RESULT and best_score > -1:
        print("\n5.3. 3D 포인트 클라우드 정합 결과 시각화 중...")
        final_transform_matrix = core.create_transform_matrix_4x4(est_tx, est_ty, est_theta_deg)
        viz.visualize_3d_registration_o3d(global_map_points_3d, live_scan_points_3d, final_transform_matrix)
        
    print("\n--- 전체 과정 완료 ---")
    
    return {
        'tx': est_tx,
        'ty': est_ty,
        'theta_deg': est_theta_deg,
        'score': best_score,
        'transform_matrix': final_transform_matrix_4x4 if best_score > -1 else None
    }

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main() 
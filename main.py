import numpy as np
import multiprocessing
import sys
import os

# HiGGSR 패키지를 찾을 수 없는 문제 해결
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from HiGGSR import core
from HiGGSR import visualization as viz

if __name__ == "__main__":
    multiprocessing.freeze_support()

    #=========================================================================
    # 설정 및 경로 정의
    #=========================================================================
    
    # 데이터 파일 경로를 HiGGSR/Data/ 하위로 변경
    BASE_DATA_PATH = os.path.join(os.path.dirname(__file__), "Data")
    GLOBAL_MAP_FILEPATH = os.path.join(BASE_DATA_PATH, "around_singong - Cloud.ply")
    LIVE_SCAN_FILEPATH = os.path.join(BASE_DATA_PATH, "around_singong_ply/001355.ply")

    #=========================================================================
    # 알고리즘 파라미터 설정
    #=========================================================================
    
    # 그리드 및 밀도 계산 설정
    GRID_CELL_SIZE = 0.20                                           
    MIN_POINTS_FOR_DENSITY_CALC = 3                                 
    DENSITY_METRIC = 'std'                                          
    KEYPOINT_DENSITY_THRESHOLD_STD = 0.1                            
    
    # 멀티프로세싱 설정
    NUM_PROCESSES = None 
    
    # 시각화 설정
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
    final_transform_dict, final_score, all_levels_visualization_data, total_hierarchical_time, total_calc_iterations = core.hierarchical_adaptive_search(
        global_keypoints, scan_keypoints,
        x_edges_global, y_edges_global, 
        LEVEL_CONFIGS,
        NUM_CANDIDATES_PER_LEVEL, 
        MIN_CANDIDATE_SEPARATION_FACTOR,
        GRID_CELL_SIZE, 
        num_processes=NUM_PROCESSES
    )

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
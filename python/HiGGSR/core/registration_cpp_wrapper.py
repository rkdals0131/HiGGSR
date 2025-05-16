# print("DEBUG: WRAPPER - FIRST LINE")
import numpy as np
import time
import sys
import os

# registration_cpp_wrapper.py의 현재 위치 (__file__)로부터 HiGGSR 프로젝트 루트를 찾음
# __file__ -> .../HiGGSR/python/HiGGSR/core/registration_cpp_wrapper.py
current_file_dir = os.path.dirname(os.path.abspath(__file__))  # .../core
higgsr_subpkg_dir = os.path.dirname(current_file_dir)          # .../HiGGSR (python/HiGGSR)
python_pkg_dir = os.path.dirname(higgsr_subpkg_dir)            # .../python
higgsr_project_root = os.path.dirname(python_pkg_dir)          # .../HiGGSR (패키지 루트)

cpp_build_dir = os.path.join(higgsr_project_root, 'cpp', 'build')

# 절대 경로로 변환
cpp_build_dir = os.path.abspath(cpp_build_dir)

# print(f"DEBUG: WRAPPER - Calculated higgsr_project_root: {higgsr_project_root}") # 경로 계산 확인용
# print(f"DEBUG: WRAPPER - Calculated cpp_build_dir: {cpp_build_dir}") # 경로 계산 확인용

if not os.path.isdir(cpp_build_dir):
    # print(f"CRITICAL: Calculated cpp_build_dir does not exist or is not a directory: {cpp_build_dir}") # 이 CRITICAL 메시지는 유지하거나, 로깅 프레임워크로 대체하는 것이 좋음
    # 기존 경로 계산 방식 (폴백 또는 디버깅용)
    fallback_cpp_build_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'cpp', 'build'))
    # print(f"DEBUG: WRAPPER - Fallback cpp_build_dir calculation was: {fallback_cpp_build_dir}") # 디버그 메시지 제거
    pass # CRITICAL 메시지는 유지하거나 필요에 따라 처리

if cpp_build_dir not in sys.path:
    sys.path.append(cpp_build_dir)

# print(f"DEBUG: WRAPPER - cpp_build_dir is {cpp_build_dir}") # 최종 사용될 경로
# print(f"DEBUG: WRAPPER - cpp_build_dir in sys.path: {cpp_build_dir in sys.path}")

# C++ 바인딩 모듈 임포트 시도
try:
    # print("DEBUG: WRAPPER - Attempting to import HiGGSR_cpp") # 디버그 메시지 제거
    import HiGGSR_cpp
    CPP_EXTENSIONS_AVAILABLE = True
    print("Successfully imported C++ extensions for HiGGSR registration (in wrapper).")
except ImportError as e:
    CPP_EXTENSIONS_AVAILABLE = False
    # 이 파일은 C++ 확장이 필수적이므로, 실패 시 에러를 발생시키거나
    # 여기서 원래 Python 구현을 다시 호출하는 것은 적절하지 않음.
    # 호출하는 쪽에서 CPP_EXTENSIONS_AVAILABLE을 확인하고 분기하도록 유도.
    print(f"CRITICAL: Could not import C++ extensions for HiGGSR registration ({e}). Wrapper will not function.") # 이 CRITICAL 메시지는 유지
    # raise ImportError(f"HiGGSR_cpp module not found. Please build the C++ extension. Error: {e}")

def call_hierarchical_adaptive_search_cpp(
    global_map_keypoints_np: np.ndarray,
    live_scan_keypoints_np: np.ndarray,
    initial_map_x_edges: list[float],
    initial_map_y_edges: list[float],
    level_configs_py: list[dict],  # Python dict list for level configurations
    num_candidates_to_select_per_level: int,
    min_candidate_separation_factor: float,
    base_grid_cell_size: float,
    num_processes: int
):
    """
    Wrapper function to call the C++ implementation of hierarchical_adaptive_search.
    Converts Python inputs to C++ compatible types and C++ results back to Python format.
    Args:
        level_configs_py: List of dictionaries, where each dictionary defines a level's parameters.
                          Example: [
                              {
                                  "grid_division": [5, 5], 
                                  "search_area_type": "full_map", # or "relative_to_map", "absolute_size"
                                  "area_param": [1.0], # float for ratio, or [width, height] for absolute
                                  "theta_range_deg": [0.0, 359.0],
                                  "theta_range_deg_relative": [-10.0, 10.0],
                                  "theta_search_steps": 36,
                                  "correspondence_distance_threshold_factor": 2.0,
                                  "tx_ty_search_steps_per_cell": [5, 5]
                              }, ...
                          ]
    Returns:
        Tuple: (final_selected_transform, final_selected_score, all_levels_viz_data, total_time_elapsed, grand_total_iterations_evaluated)
               This matches the return signature of the original Python version.
               all_levels_viz_data is currently returned as an empty list as C++ version doesn't produce it yet.
    """
    if not CPP_EXTENSIONS_AVAILABLE:
        # C++ 모듈이 로드되지 않았으면, 여기서 에러를 발생시키거나 None을 반환하여
        # 호출 측에서 원래 Python 구현을 사용하도록 할 수 있습니다.
        # 여기서는 예외를 발생시켜 C++ 확장이 필수임을 명확히 합니다.
        raise RuntimeError("HiGGSR C++ extensions are not available. Cannot call C++ implementation.")

    print("\nUsing C++ implementation via registration_cpp_wrapper.")

    # 1. Convert Python level_configs (list of dicts) to list of C++ LevelConfig objects
    cpp_level_configs = []
    for py_config in level_configs_py:
        cfg = HiGGSR_cpp.LevelConfig()  # Access LevelConfig class from the C++ module
        
        cfg.grid_division = py_config.get("grid_division", [5, 5])
        cfg.search_area_type = py_config.get("search_area_type", "full_map")
        
        area_param_py = py_config.get("area_param") # Python side: float or list/tuple
        if isinstance(area_param_py, (int, float)):
            cfg.area_param = [float(area_param_py)]
        elif isinstance(area_param_py, (list, tuple)):
            cfg.area_param = [float(x) for x in area_param_py]
        else:
            # Default based on C++ constructor or logic here
            if cfg.search_area_type == "full_map":
                cfg.area_param = [1.0] 
            elif cfg.search_area_type == "absolute_size":
                # Provide a sensible default if not specified for absolute_size
                cfg.area_param = [20.0 * base_grid_cell_size, 20.0 * base_grid_cell_size] 
            else: # relative_to_map
                cfg.area_param = [0.1] # Default ratio

        cfg.theta_range_deg = py_config.get("theta_range_deg", [0.0, 359.0])
        cfg.theta_range_deg_relative = py_config.get("theta_range_deg_relative", [-10.0, 10.0])
        cfg.theta_search_steps = py_config.get("theta_search_steps", 36)
        cfg.correspondence_distance_threshold_factor = py_config.get("correspondence_distance_threshold_factor", 2.0)
        cfg.tx_ty_search_steps_per_cell = py_config.get("tx_ty_search_steps_per_cell", [5, 5])
        cpp_level_configs.append(cfg)

    # 2. Create C++ Registration object
    cpp_registration = HiGGSR_cpp.Registration()

    # 3. Ensure NumPy arrays are C-contiguous and float32 for C++ extension
    # The pybind11 bindings already specify py::array::c_style | py::array::forcecast,
    # but explicit conversion can sometimes avoid subtle issues.
    gmk_np_cpp = np.ascontiguousarray(global_map_keypoints_np, dtype=np.float32)
    lsk_np_cpp = np.ascontiguousarray(live_scan_keypoints_np, dtype=np.float32)

    # initial_map_x_edges and initial_map_y_edges should be list[float]
    # pybind11 automatically converts Python list to std::vector<float>.
    initial_x_edges_list = list(initial_map_x_edges)
    initial_y_edges_list = list(initial_map_y_edges)

    # 4. Call the C++ method
    print(f"  Invoking C++ hierarchicalAdaptiveSearch from wrapper...")
    start_time_cpp_call = time.time()
    result_cpp = cpp_registration.hierarchical_adaptive_search(
        gmk_np_cpp,
        lsk_np_cpp,
        initial_x_edges_list,
        initial_y_edges_list,
        cpp_level_configs,
        num_candidates_to_select_per_level,
        min_candidate_separation_factor,
        base_grid_cell_size,
        num_processes
    )
    end_time_cpp_call = time.time()
    cpp_call_duration = end_time_cpp_call - start_time_cpp_call

    # 5. Convert C++ RegistrationResult back to Python-friendly format
    # The `transform` member of RegistrationResult is an Eigen::Matrix4f,
    # which pybind11 converts to a NumPy array automatically.
    final_transform_matrix_cpp = result_cpp.transform

    # Extract tx, ty, theta_deg from the transformation matrix
    tx_cpp = final_transform_matrix_cpp[0, 3]
    ty_cpp = final_transform_matrix_cpp[1, 3]
    
    # Calculate theta_deg (rotation around Z from 2D rotation matrix part)
    # R = [[cos(th), -sin(th)], [sin(th), cos(th)]]
    # So, R(0,0) = cos(th), R(1,0) = sin(th)
    theta_rad_cpp = np.arctan2(final_transform_matrix_cpp[1, 0], final_transform_matrix_cpp[0, 0])
    theta_deg_cpp = np.rad2deg(theta_rad_cpp)

    final_selected_transform_from_cpp = {
        'tx': tx_cpp,
        'ty': ty_cpp,
        'theta_deg': theta_deg_cpp,
        'score': result_cpp.score
    }

    # Visualization data (all_levels_viz_data) is currently not produced by the C++ implementation.
    # To match the Python version's output, C++ would need to generate and return this.
    # For now, returning an empty list as a placeholder.
    all_levels_viz_data_cpp = [] 

    print(f"  C++ (hierarchicalAdaptiveSearch) internal time: {result_cpp.time_elapsed_sec:.3f} s")
    print(f"  C++ wrapper call overhead + C++ internal time: {cpp_call_duration:.3f} s")
    print(f"  C++ Total Iterations: {result_cpp.total_iterations}")
    print(f"  C++ Best Result: tx={tx_cpp:.3f}, ty={ty_cpp:.3f}, th={theta_deg_cpp:.2f} deg, score={result_cpp.score}")

    return (
        final_selected_transform_from_cpp, 
        result_cpp.score, 
        all_levels_viz_data_cpp, 
        result_cpp.time_elapsed_sec,  # Using C++ internal time measurement
        result_cpp.total_iterations
    )

# Example of how this wrapper might be used from another Python script:
if __name__ == '__main__':
    # This is just a placeholder for example usage.
    # Actual data and level_configs would need to be defined.
    print("Example usage of call_hierarchical_adaptive_search_cpp:")
    
    if CPP_EXTENSIONS_AVAILABLE:
        # Dummy data for testing
        dummy_points = np.random.rand(100, 3).astype(np.float32)
        dummy_map_edges = [0.0, 10.0]
        dummy_level_configs = [
            {
                "grid_division": [2, 2],
                "search_area_type": "full_map",
                "theta_range_deg": [0, 90],
                "theta_search_steps": 4,
                "correspondence_distance_threshold_factor": 1.5,
                "tx_ty_search_steps_per_cell": [3,3]
            }
        ]
        try:
            result = call_hierarchical_adaptive_search_cpp(
                global_map_keypoints_np=dummy_points,
                live_scan_keypoints_np=dummy_points, # Using same for simplicity
                initial_map_x_edges=dummy_map_edges,
                initial_map_y_edges=dummy_map_edges,
                level_configs_py=dummy_level_configs,
                num_candidates_to_select_per_level=1,
                min_candidate_separation_factor=1.0,
                base_grid_cell_size=0.5,
                num_processes=0
            )
            print("\nWrapper call successful. Result:")
            print(f"  Transform: {result[0]}")
            print(f"  Score: {result[1]}")
            print(f"  Time (C++): {result[3]}")
            print(f"  Iterations (C++): {result[4]}")

        except RuntimeError as e:
            print(f"Error during example C++ call: {e}")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
    else:
        print("C++ extensions not available, cannot run example.") 
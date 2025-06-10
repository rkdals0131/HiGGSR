"""
Registration C++ Wrapper

복잡한 계층적 적응 탐색을 위한 C++ 래퍼
C++ 구현이 사용 가능할 때는 C++ 함수를 호출하고,
그렇지 않으면 Python 구현으로 Fallback하는 래퍼
"""

import numpy as np
from typing import Dict, List, Union, Tuple, Any, Optional
import warnings

# C++ 모듈 임포트 시도
try:
    from higgsr_ros.core import higgsr_core_cpp
    CPP_AVAILABLE = True
except ImportError:
    higgsr_core_cpp = None
    CPP_AVAILABLE = False

# Python 구현 임포트 (Fallback용)
from higgsr_ros.core.registration import (
    hierarchical_adaptive_search as hierarchical_adaptive_search_python,
    count_correspondences_kdtree as count_correspondences_kdtree_python,
    select_diverse_candidates as select_diverse_candidates_python
)


def _validate_hierarchical_search_inputs(
    global_map_keypoints: np.ndarray,
    live_scan_keypoints: np.ndarray,
    initial_map_x_edges: np.ndarray,
    initial_map_y_edges: np.ndarray,
    level_configs: List[Dict[str, Any]]
) -> None:
    """
    계층적 탐색 입력 파라미터 유효성 검증
    
    Args:
        global_map_keypoints: 글로벌 맵 키포인트 (N x 2)
        live_scan_keypoints: 현재 스캔 키포인트 (M x 2)
        initial_map_x_edges: 초기 맵 X 경계값
        initial_map_y_edges: 초기 맵 Y 경계값
        level_configs: 레벨 설정 리스트
        
    Raises:
        TypeError: 타입이 올바르지 않은 경우
        ValueError: 값이 유효하지 않은 경우
        AttributeError: 필수 속성이 없는 경우
    """
    # 타입 검증
    if not isinstance(global_map_keypoints, np.ndarray):
        raise TypeError("global_map_keypoints must be numpy.ndarray")
    if not isinstance(live_scan_keypoints, np.ndarray):
        raise TypeError("live_scan_keypoints must be numpy.ndarray")
    if not isinstance(initial_map_x_edges, np.ndarray):
        raise TypeError("initial_map_x_edges must be numpy.ndarray")
    if not isinstance(initial_map_y_edges, np.ndarray):
        raise TypeError("initial_map_y_edges must be numpy.ndarray")
    if not isinstance(level_configs, list):
        raise TypeError("level_configs must be list")
    
    # 형태 검증
    if global_map_keypoints.ndim != 2 or global_map_keypoints.shape[1] != 2:
        raise ValueError("global_map_keypoints must be N x 2 array")
    if live_scan_keypoints.ndim != 2 or live_scan_keypoints.shape[1] != 2:
        raise ValueError("live_scan_keypoints must be M x 2 array")
    if initial_map_x_edges.ndim != 1:
        raise ValueError("initial_map_x_edges must be 1-dimensional")
    if initial_map_y_edges.ndim != 1:
        raise ValueError("initial_map_y_edges must be 1-dimensional")
    
    # 크기 검증
    if global_map_keypoints.size == 0:
        raise ValueError("global_map_keypoints cannot be empty")
    if live_scan_keypoints.size == 0:
        raise ValueError("live_scan_keypoints cannot be empty")
    if initial_map_x_edges.size < 2:
        raise ValueError("initial_map_x_edges must have at least 2 elements")
    if initial_map_y_edges.size < 2:
        raise ValueError("initial_map_y_edges must have at least 2 elements")
    if len(level_configs) == 0:
        raise ValueError("level_configs cannot be empty")
    
    # level_configs 구조 검증
    for i, config in enumerate(level_configs):
        if not isinstance(config, dict):
            raise TypeError(f"level_configs[{i}] must be dict")
        
        # 필수 키 확인
        required_keys = ['grid_division', 'search_area_type', 'tx_ty_search_steps']
        for key in required_keys:
            if key not in config:
                raise AttributeError(f"level_configs[{i}] missing required key: {key}")
        
        # 타입 확인
        if not isinstance(config['grid_division'], (list, tuple)):
            raise TypeError(f"level_configs[{i}]['grid_division'] must be list or tuple")
        if len(config['grid_division']) != 2:
            raise ValueError(f"level_configs[{i}]['grid_division'] must have 2 elements")
        if not all(isinstance(x, int) and x > 0 for x in config['grid_division']):
            raise ValueError(f"level_configs[{i}]['grid_division'] must contain positive integers")
        
        if not isinstance(config['search_area_type'], str):
            raise TypeError(f"level_configs[{i}]['search_area_type'] must be string")
        
        if not isinstance(config['tx_ty_search_steps'], (list, tuple)):
            raise TypeError(f"level_configs[{i}]['tx_ty_search_steps'] must be list or tuple")
        if len(config['tx_ty_search_steps']) != 2:
            raise ValueError(f"level_configs[{i}]['tx_ty_search_steps'] must have 2 elements")
        if not all(isinstance(x, int) and x > 0 for x in config['tx_ty_search_steps']):
            raise ValueError(f"level_configs[{i}]['tx_ty_search_steps'] must contain positive integers")
    
    # 경계값 정렬 확인
    if not np.all(np.diff(initial_map_x_edges) > 0):
        raise ValueError("initial_map_x_edges must be in ascending order")
    if not np.all(np.diff(initial_map_y_edges) > 0):
        raise ValueError("initial_map_y_edges must be in ascending order")
    
    # NaN/Inf 검증
    if not np.all(np.isfinite(global_map_keypoints)):
        raise ValueError("global_map_keypoints contains non-finite values")
    if not np.all(np.isfinite(live_scan_keypoints)):
        raise ValueError("live_scan_keypoints contains non-finite values")
    if not np.all(np.isfinite(initial_map_x_edges)):
        raise ValueError("initial_map_x_edges contains non-finite values")
    if not np.all(np.isfinite(initial_map_y_edges)):
        raise ValueError("initial_map_y_edges contains non-finite values")


def _convert_result_to_dict(result: Any) -> Dict[str, Union[float, int, bool]]:
    """
    C++ 결과 객체를 Python dict로 변환
    
    Args:
        result: C++ TransformResult 객체 또는 Python dict
        
    Returns:
        Dict: 변환된 결과 딕셔너리
    """
    if isinstance(result, dict):
        return result
    
    # C++ 객체에서 속성 추출
    try:
        return {
            'tx': float(result.tx),
            'ty': float(result.ty),
            'theta_deg': float(result.theta_deg),
            'score': float(result.score),
            'iterations': int(result.iterations),
            'success': bool(result.success)
        }
    except AttributeError as e:
        raise RuntimeError(f"Invalid result object: {e}")


def hierarchical_adaptive_search_cpp(
    global_map_keypoints: np.ndarray,
    live_scan_keypoints: np.ndarray,
    initial_map_x_edges: np.ndarray,
    initial_map_y_edges: np.ndarray,
    level_configs: List[Dict[str, Any]],
    num_candidates_to_select_per_level: int = 5,
    min_candidate_separation_factor: float = 2.0,
    base_grid_cell_size: float = 1.0,
    num_processes: int = 0,
    use_cpp: bool = True
) -> Dict[str, Union[float, int, bool]]:
    """
    C++ 가속화된 계층적 적응 탐색 함수
    
    Args:
        global_map_keypoints: 글로벌 맵의 키포인트들 (N x 2)
        live_scan_keypoints: 현재 스캔의 키포인트들 (M x 2)
        initial_map_x_edges: 초기 맵 X 경계값 배열
        initial_map_y_edges: 초기 맵 Y 경계값 배열
        level_configs: 각 레벨의 탐색 설정 리스트
        num_candidates_to_select_per_level: 레벨당 선택할 후보 수
        min_candidate_separation_factor: 최소 후보 분리 팩터
        base_grid_cell_size: 기본 그리드 셀 크기
        num_processes: 병렬 처리 프로세스 수 (0은 시퀀셜)
        use_cpp: C++ 구현 사용 여부
        
    Returns:
        Dict: 최적 변환 결과
            - tx: X 방향 이동
            - ty: Y 방향 이동
            - theta_deg: 회전 각도(도)
            - score: 매칭 점수
            - iterations: 수행된 반복 횟수
            - success: 성공 여부
            
    Raises:
        TypeError: 입력 타입이 올바르지 않은 경우
        ValueError: 입력 값이 유효하지 않은 경우
        RuntimeError: 탐색 중 오류가 발생한 경우
    """
    # 입력 유효성 검증
    _validate_hierarchical_search_inputs(
        global_map_keypoints, live_scan_keypoints,
        initial_map_x_edges, initial_map_y_edges, level_configs
    )
    
    # 추가 파라미터 검증
    if not isinstance(num_candidates_to_select_per_level, int) or num_candidates_to_select_per_level <= 0:
        raise ValueError("num_candidates_to_select_per_level must be positive integer")
    if not isinstance(min_candidate_separation_factor, (int, float)) or min_candidate_separation_factor <= 0:
        raise ValueError("min_candidate_separation_factor must be positive number")
    if not isinstance(base_grid_cell_size, (int, float)) or base_grid_cell_size <= 0:
        raise ValueError("base_grid_cell_size must be positive number")
    if not isinstance(num_processes, int) or num_processes < 0:
        raise ValueError("num_processes must be non-negative integer")
    
    try:
        # C++ 구현 사용 시도
        if use_cpp and CPP_AVAILABLE and higgsr_core_cpp is not None:
            # 타입 변환
            global_keypoints_double = global_map_keypoints.astype(np.float64, copy=False)
            scan_keypoints_double = live_scan_keypoints.astype(np.float64, copy=False)
            x_edges_double = initial_map_x_edges.astype(np.float64, copy=False)
            y_edges_double = initial_map_y_edges.astype(np.float64, copy=False)
            
            result = higgsr_core_cpp.hierarchical_adaptive_search(
                global_keypoints_double,
                scan_keypoints_double,
                x_edges_double,
                y_edges_double,
                level_configs,
                num_candidates_to_select_per_level,
                min_candidate_separation_factor,
                base_grid_cell_size,
                num_processes
            )
            
            # 결과 변환
            result_dict = _convert_result_to_dict(result)
            
            # 결과 검증
            required_keys = ['tx', 'ty', 'theta_deg', 'score', 'iterations', 'success']
            for key in required_keys:
                if key not in result_dict:
                    raise RuntimeError(f"C++ function returned incomplete result: missing {key}")
            
            return result_dict
            
    except Exception as e:
        if use_cpp:
            warnings.warn(f"C++ implementation failed, falling back to Python: {e}")
    
    # Python 구현으로 Fallback
    try:
        result = hierarchical_adaptive_search_python(
            global_map_keypoints,
            live_scan_keypoints,
            initial_map_x_edges,
            initial_map_y_edges,
            level_configs,
            num_candidates_to_select_per_level,
            min_candidate_separation_factor,
            base_grid_cell_size,
            num_processes
        )
        
        # Python 결과를 dict 형태로 변환 (필요시)
        if not isinstance(result, dict):
            # Python 함수가 튜플이나 다른 형태로 반환하는 경우 처리
            if hasattr(result, '_asdict'):  # namedtuple인 경우
                result = result._asdict()
            else:
                raise RuntimeError("Python function returned unexpected result type")
        
        return result
        
    except Exception as e:
        raise RuntimeError(f"Both C++ and Python implementations failed: {e}")


def count_correspondences_kdtree_cpp(
    transformed_keypoints: np.ndarray,
    global_map_keypoints: np.ndarray,
    distance_threshold: float,
    use_cpp: bool = True
) -> int:
    """
    C++ 가속화된 KDTree 기반 대응점 계산 함수
    
    Args:
        transformed_keypoints: 변환된 키포인트들 (N x 2)
        global_map_keypoints: 글로벌 맵 키포인트들 (M x 2)
        distance_threshold: 매칭 거리 임계값
        use_cpp: C++ 구현 사용 여부
        
    Returns:
        int: 매칭된 대응점 개수
        
    Raises:
        TypeError: 입력 타입이 올바르지 않은 경우
        ValueError: 입력 값이 유효하지 않은 경우
    """
    # 입력 유효성 검증
    if not isinstance(transformed_keypoints, np.ndarray):
        raise TypeError("transformed_keypoints must be numpy.ndarray")
    if not isinstance(global_map_keypoints, np.ndarray):
        raise TypeError("global_map_keypoints must be numpy.ndarray")
    if not isinstance(distance_threshold, (int, float)):
        raise TypeError("distance_threshold must be numeric")
    
    if transformed_keypoints.ndim != 2 or transformed_keypoints.shape[1] != 2:
        raise ValueError("transformed_keypoints must be N x 2 array")
    if global_map_keypoints.ndim != 2 or global_map_keypoints.shape[1] != 2:
        raise ValueError("global_map_keypoints must be M x 2 array")
    if distance_threshold <= 0 or not np.isfinite(distance_threshold):
        raise ValueError("distance_threshold must be positive and finite")
    
    # 빈 배열 처리
    if transformed_keypoints.size == 0 or global_map_keypoints.size == 0:
        return 0
    
    try:
        # C++ 구현 사용 시도
        if use_cpp and CPP_AVAILABLE and higgsr_core_cpp is not None:
            # 타입 변환
            transformed_double = transformed_keypoints.astype(np.float64, copy=False)
            global_double = global_map_keypoints.astype(np.float64, copy=False)
            
            result = higgsr_core_cpp.count_correspondences_kdtree(
                transformed_double,
                global_double,
                float(distance_threshold)
            )
            
            if not isinstance(result, int) or result < 0:
                raise RuntimeError("C++ function returned invalid correspondence count")
                
            return result
            
    except Exception as e:
        if use_cpp:
            warnings.warn(f"C++ implementation failed, falling back to Python: {e}")
    
    # Python 구현으로 Fallback
    try:
        # Python 함수 호출시 KDTree 객체가 필요할 수 있음
        from scipy.spatial import KDTree
        global_kdtree = KDTree(global_map_keypoints)
        
        result = count_correspondences_kdtree_python(
            transformed_keypoints, global_kdtree, distance_threshold
        )
        
        return int(result)
        
    except Exception as e:
        raise RuntimeError(f"Both C++ and Python implementations failed: {e}")


# 편의 함수들 (기존 인터페이스 호환성 유지)
def hierarchical_adaptive_search(
    global_map_keypoints: np.ndarray,
    live_scan_keypoints: np.ndarray,
    initial_map_x_edges: np.ndarray,
    initial_map_y_edges: np.ndarray,
    level_configs: List[Dict[str, Any]],
    num_candidates_to_select_per_level: int = 5,
    min_candidate_separation_factor: float = 2.0,
    base_grid_cell_size: float = 1.0,
    num_processes: int = 0
) -> Dict[str, Union[float, int, bool]]:
    """기존 함수명 호환성을 위한 래퍼"""
    return hierarchical_adaptive_search_cpp(
        global_map_keypoints, live_scan_keypoints,
        initial_map_x_edges, initial_map_y_edges, level_configs,
        num_candidates_to_select_per_level, min_candidate_separation_factor,
        base_grid_cell_size, num_processes, use_cpp=True
    )


def count_correspondences_kdtree(
    transformed_keypoints: np.ndarray,
    global_map_kdtree: Any,  # KDTree 객체 또는 키포인트 배열
    distance_threshold: float
) -> int:
    """기존 함수명 호환성을 위한 래퍼"""
    # KDTree 객체인지 키포인트 배열인지 판단
    if hasattr(global_map_kdtree, 'query'):  # KDTree 객체
        # KDTree에서 데이터 추출
        global_map_keypoints = global_map_kdtree.data
    else:  # 키포인트 배열
        global_map_keypoints = global_map_kdtree
    
    return count_correspondences_kdtree_cpp(
        transformed_keypoints, global_map_keypoints, distance_threshold, use_cpp=True
    )


# TODO: 향후 추가될 기능들
# def select_diverse_candidates_cpp(candidates_info, num_to_select, separation_factor, 
#                                  cell_size_x, cell_size_y, map_x_range, map_y_range, use_cpp=True):
#     """다양한 후보 선택 (C++ 가속화)"""
#     pass
#
# def parallel_search_in_super_grids_cpp(global_keypoints, scan_keypoints, config, num_threads, use_cpp=True):
#     """병렬 그리드 탐색 (OpenMP 활용)"""
#     pass 
# HiGGSR Core 모듈 초기화
# C++ 확장 모듈과 Python 구현 간의 동적 스위칭 지원

print("INFO: Initializing HiGGSR Core module...")

# 기본 유틸리티 함수들 (항상 사용 가능)
from .utils import load_point_cloud_from_file, create_2d_height_variance_map, create_transform_matrix_4x4

# C++ 확장 모듈 사용 가능 여부 확인
CPP_EXTENSIONS_AVAILABLE = False
USE_CPP_EXTENSIONS = False

try:
    # C++ 래퍼 모듈 임포트 시도
    from .cpp_wrappers import CPP_EXTENSIONS_AVAILABLE as cpp_available
    if cpp_available:
        from .cpp_wrappers.feature_extraction_wrapper import (
            extract_high_density_keypoints as extract_high_density_keypoints_cpp,
            apply_transform_to_keypoints_numba as apply_transform_to_keypoints_cpp
        )
        from .cpp_wrappers.registration_wrapper import (
            hierarchical_adaptive_search as hierarchical_adaptive_search_cpp,
            count_correspondences_kdtree as count_correspondences_kdtree_cpp
        )
        CPP_EXTENSIONS_AVAILABLE = True
        print("INFO: C++ acceleration modules successfully loaded")
    else:
        print("INFO: C++ modules not available, using Python implementation")
        
except ImportError as e:
    print(f"INFO: C++ wrapper import failed: {e}")
    print("INFO: Falling back to Python implementation")

# Python 구현 임포트 (Fallback 및 기본 구현)
from .feature_extraction import (
    extract_high_density_keypoints as extract_high_density_keypoints_python,
    apply_transform_to_keypoints_numba as apply_transform_to_keypoints_python
)
from .registration import (
    hierarchical_adaptive_search as hierarchical_adaptive_search_python,
    count_correspondences_kdtree as count_correspondences_kdtree_python,
    search_in_super_grids,
    select_diverse_candidates
)

# 사용할 구현 결정 (기본적으로 Python 사용, 명시적 활성화 필요)
def set_use_cpp_extensions(enable: bool) -> None:
    """
    C++ 확장 사용 여부를 설정하는 함수
    
    Args:
        enable: C++ 확장 사용 여부
    """
    global USE_CPP_EXTENSIONS
    if enable and not CPP_EXTENSIONS_AVAILABLE:
        print("WARNING: C++ extensions requested but not available. Using Python implementation.")
        USE_CPP_EXTENSIONS = False
    else:
        USE_CPP_EXTENSIONS = enable
        if enable:
            print("INFO: C++ acceleration enabled")
        else:
            print("INFO: Using Python implementation")
    
    _update_function_references()

def _update_function_references() -> None:
    """현재 설정에 따라 함수 참조를 업데이트"""
    global extract_high_density_keypoints, apply_transform_to_keypoints_numba, hierarchical_adaptive_search
    
    if USE_CPP_EXTENSIONS and CPP_EXTENSIONS_AVAILABLE:
        # C++ 가속화 버전 사용
        extract_high_density_keypoints = extract_high_density_keypoints_cpp
        apply_transform_to_keypoints_numba = apply_transform_to_keypoints_cpp  
        hierarchical_adaptive_search = hierarchical_adaptive_search_cpp
        print("INFO: Using C++ accelerated implementations")
    else:
        # Python 구현 사용
        extract_high_density_keypoints = extract_high_density_keypoints_python
        apply_transform_to_keypoints_numba = apply_transform_to_keypoints_python
        hierarchical_adaptive_search = hierarchical_adaptive_search_python
        print("INFO: Using Python implementations")

# 초기 함수 참조 설정 (기본값: Python)
extract_high_density_keypoints = None
apply_transform_to_keypoints_numba = None  
hierarchical_adaptive_search = None

# 초기화 - C++ 확장이 사용 가능하면 자동으로 활성화
if CPP_EXTENSIONS_AVAILABLE:
    set_use_cpp_extensions(True)
else:
    _update_function_references()

# 시스템 정보 출력
def print_acceleration_status() -> None:
    """현재 가속화 상태를 출력"""
    print(f"=== HiGGSR Core Acceleration Status ===")
    print(f"C++ Extensions Available: {CPP_EXTENSIONS_AVAILABLE}")
    print(f"Currently Using C++: {USE_CPP_EXTENSIONS}")
    if USE_CPP_EXTENSIONS:
        print(f"Active Implementation: C++ Accelerated")
    else:
        print(f"Active Implementation: Python (Numba JIT)")
    print(f"=====================================")

# 호환성을 위한 추가 함수들
def get_cpp_acceleration_status() -> dict:
    """C++ 가속화 상태 정보를 반환"""
    return {
        'cpp_available': CPP_EXTENSIONS_AVAILABLE,
        'cpp_enabled': USE_CPP_EXTENSIONS,
        'current_backend': 'cpp' if USE_CPP_EXTENSIONS else 'python'
    }

# TODO: 개발 중인 기능들 플래그
DEVELOPMENT_FEATURES = {
    'pcl_integration': False,  # PCL 통합 기능
    'openmp_parallel': False,  # OpenMP 병렬화
    'gpu_acceleration': False,  # GPU 가속화 (향후)
}

# 모듈 익스포트
__all__ = [
    # 유틸리티 함수들
    'load_point_cloud_from_file', 
    'create_2d_height_variance_map',
    'create_transform_matrix_4x4',
    
    # 동적으로 할당되는 핵심 함수들
    'extract_high_density_keypoints',
    'apply_transform_to_keypoints_numba',
    'hierarchical_adaptive_search',
    
    # Registration 보조 함수들 (Python 전용)
    'search_in_super_grids',
    'count_correspondences_kdtree',
    'select_diverse_candidates',
    
    # 설정 및 상태 함수들
    'set_use_cpp_extensions',
    'print_acceleration_status', 
    'get_cpp_acceleration_status',
    
    # 상수들
    'CPP_EXTENSIONS_AVAILABLE',
    'USE_CPP_EXTENSIONS',
    'DEVELOPMENT_FEATURES'
]

print("INFO: HiGGSR Core module initialization complete") 
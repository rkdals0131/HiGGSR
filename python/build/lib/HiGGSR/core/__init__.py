print("DEBUG: CORE INIT - FIRST LINE")
from .utils import load_point_cloud_from_file, create_2d_height_variance_map, create_transform_matrix_4x4
from .feature_extraction import extract_high_density_keypoints, apply_transform_to_keypoints_numba

# C++ Wrapper와 Python 구현 임포트
try:
    # registration_cpp_wrapper 내부에서 HiGGSR_cpp 임포트 시도 및 CPP_EXTENSIONS_AVAILABLE 설정
    from .registration_cpp_wrapper import call_hierarchical_adaptive_search_cpp, CPP_EXTENSIONS_AVAILABLE
except ImportError: # registration_cpp_wrapper.py 파일 자체가 없거나 임포트 중 다른 에러 발생 시
    CPP_EXTENSIONS_AVAILABLE = False
    call_hierarchical_adaptive_search_cpp = None 
    print("CRITICAL: Failed to import 'registration_cpp_wrapper'. C++ extensions will not be used.")

# 순수 Python 구현 임포트 (registration.py 내부 함수들을 직접 가져옴)
from .registration import hierarchical_adaptive_search as hierarchical_adaptive_search_python
from .registration import (
    search_in_super_grids,
    count_correspondences_kdtree,
    select_diverse_candidates
)

# C++ 확장 사용 가능 여부에 따라 hierarchical_adaptive_search 함수 선택
print(f"DEBUG: CORE INIT - CPP_EXTENSIONS_AVAILABLE is {CPP_EXTENSIONS_AVAILABLE}")
print(f"DEBUG: CORE INIT - call_hierarchical_adaptive_search_cpp is not None: {call_hierarchical_adaptive_search_cpp is not None}")
if CPP_EXTENSIONS_AVAILABLE and call_hierarchical_adaptive_search_cpp is not None:
    hierarchical_adaptive_search = call_hierarchical_adaptive_search_cpp
    # 성공 메시지는 registration_cpp_wrapper에서 이미 출력됨
    print("INFO: HiGGSR core will use C++ accelerated registration.")
elif call_hierarchical_adaptive_search_cpp is None and CPP_EXTENSIONS_AVAILABLE:
    # 이 경우는 registration_cpp_wrapper는 임포트 되었으나, 내부에서 HiGGSR_cpp 모듈 로드 실패한 경우.
    # 해당 wrapper 파일에서 이미 CRITICAL 메시지를 출력했을 것임.
    hierarchical_adaptive_search = hierarchical_adaptive_search_python
    print("WARNING: HiGGSR core falling back to Python (wrapper imported, C++ module failed).")
else:
    # CPP_EXTENSIONS_AVAILABLE가 False인 모든 다른 경우 (wrapper 임포트 실패 포함)
    hierarchical_adaptive_search = hierarchical_adaptive_search_python
    print("WARNING: HiGGSR core falling back to Python (C++ extensions not available or wrapper import failed).")

__all__ = [
    'load_point_cloud_from_file', 
    'create_2d_height_variance_map',
    'create_transform_matrix_4x4',
    'extract_high_density_keypoints',
    'apply_transform_to_keypoints_numba',
    'search_in_super_grids',
    'hierarchical_adaptive_search', # 최종 선택된 함수
    'count_correspondences_kdtree',
    'select_diverse_candidates'
] 
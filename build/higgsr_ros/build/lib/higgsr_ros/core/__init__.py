# print("DEBUG: CORE INIT - FIRST LINE")
from .utils import load_point_cloud_from_file, create_2d_height_variance_map, create_transform_matrix_4x4
from .feature_extraction import extract_high_density_keypoints, apply_transform_to_keypoints_numba

# C++ Wrapper와 Python 구현 임포트
# try:
#     # registration_cpp_wrapper 내부에서 HiGGSR_cpp 임포트 시도 및 CPP_EXTENSIONS_AVAILABLE 설정
#     from .registration_cpp_wrapper import call_hierarchical_adaptive_search_cpp, CPP_EXTENSIONS_AVAILABLE as CPP_MODULE_LOADED
# except ImportError: # registration_cpp_wrapper.py 파일 자체가 없거나 임포트 중 다른 에러 발생 시
#     CPP_MODULE_LOADED = False
#     call_hierarchical_adaptive_search_cpp = None 
#     print("CRITICAL: Failed to import \'registration_cpp_wrapper\'. C++ extensions will not be used.")
CPP_MODULE_LOADED = False
call_hierarchical_adaptive_search_cpp = None

# 순수 Python 구현 임포트 (registration.py 내부 함수들을 직접 가져옴)
from .registration import hierarchical_adaptive_search as hierarchical_adaptive_search_python
from .registration import (
    search_in_super_grids,
    count_correspondences_kdtree,
    select_diverse_candidates
)

# C++ 확장 사용 여부 플래그 (main.py에서 설정)
_USE_CPP_BY_USER_SETTING = False # 기본적으로 Python 사용

def set_use_cpp_extensions(should_use: bool):
    """
    main.py에서 사용자가 C++ 확장 사용 여부를 설정할 수 있도록 하는 함수.
    ROS2에서는 이 함수가 호출되지 않거나, 항상 False로 설정되도록 할 예정.
    """
    global _USE_CPP_BY_USER_SETTING
    # _USE_CPP_BY_USER_SETTING = should_use # C++ 사용 안함
    _USE_CPP_BY_USER_SETTING = False
    print(f"INFO: set_use_cpp_extensions called with {should_use}, but C++ extensions are disabled for ROS 2 version.")
    _select_implementation()


# 실제 사용할 함수 (초기에는 None 또는 Python 구현으로 설정 후 _select_implementation에서 결정)
hierarchical_adaptive_search = None

def _select_implementation():
    global hierarchical_adaptive_search
    # print(f"DEBUG: CORE INIT - CPP_MODULE_LOADED is {CPP_MODULE_LOADED}")
    # print(f"DEBUG: CORE INIT - call_hierarchical_adaptive_search_cpp is not None: {call_hierarchical_adaptive_search_cpp is not None}")
    # print(f"DEBUG: CORE INIT - _USE_CPP_BY_USER_SETTING is {_USE_CPP_BY_USER_SETTING}")

    # if _USE_CPP_BY_USER_SETTING and CPP_MODULE_LOADED and call_hierarchical_adaptive_search_cpp is not None:
    #     hierarchical_adaptive_search = call_hierarchical_adaptive_search_cpp
    #     # 성공 메시지는 registration_cpp_wrapper에서 이미 출력됨
    #     print("INFO: HiGGSR core will use C++ accelerated registration.")
    # elif _USE_CPP_BY_USER_SETTING and call_hierarchical_adaptive_search_cpp is None and CPP_MODULE_LOADED:
    #     # 이 경우는 registration_cpp_wrapper는 임포트 되었으나, 내부에서 HiGGSR_cpp 모듈 로드 실패한 경우.
    #     # 해당 wrapper 파일에서 이미 CRITICAL 메시지를 출력했을 것임.
    #     hierarchical_adaptive_search = hierarchical_adaptive_search_python
    #     print("WARNING: HiGGSR core falling back to Python (wrapper imported, C++ module failed, user wanted C++).")
    # elif _USE_CPP_BY_USER_SETTING and not CPP_MODULE_LOADED:
    #     hierarchical_adaptive_search = hierarchical_adaptive_search_python
    #     print("WARNING: HiGGSR core falling back to Python (C++ module or wrapper not available, user wanted C++).")
    # else: # not _USE_CPP_BY_USER_SETTING or other unhandled cases
    #     hierarchical_adaptive_search = hierarchical_adaptive_search_python
    #     if not _USE_CPP_BY_USER_SETTING:
    #         print("INFO: HiGGSR core will use Python implementation (user-selected).")
    #     else:
    #         print("WARNING: HiGGSR core falling back to Python (C++ extensions not available or wrapper import failed).")
    hierarchical_adaptive_search = hierarchical_adaptive_search_python
    print("INFO: HiGGSR core (ROS 2) will use Python implementation.")

_select_implementation() # 모듈 로드 시 함수 선택 실행


__all__ = [
    'load_point_cloud_from_file', 
    'create_2d_height_variance_map',
    'create_transform_matrix_4x4',
    'extract_high_density_keypoints',
    'apply_transform_to_keypoints_numba',
    'search_in_super_grids',
    'hierarchical_adaptive_search', # 최종 선택된 함수
    'count_correspondences_kdtree',
    'select_diverse_candidates',
    'set_use_cpp_extensions' # 외부에서 호출 가능하도록 export
] 
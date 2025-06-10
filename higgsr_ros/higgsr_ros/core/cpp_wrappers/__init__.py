# C++ Wrappers 모듈 초기화
# C++ 모듈 로딩과 Fallback 로직을 처리

__all__ = [
    'feature_extraction_wrapper',
    'registration_wrapper',
    'CPP_EXTENSIONS_AVAILABLE'
]

# C++ 확장 모듈 사용 가능 여부
CPP_EXTENSIONS_AVAILABLE = False

try:
    # C++ 모듈 임포트 시도
    import higgsr_ros.core.higgsr_core_cpp as higgsr_core_cpp
    CPP_EXTENSIONS_AVAILABLE = True
    print("INFO: C++ extensions successfully loaded")
except ImportError as e:
    print(f"INFO: C++ extensions not available: {e}")
    higgsr_core_cpp = None

# 래퍼 모듈들 임포트
try:
    from . import feature_extraction_wrapper
    from . import registration_wrapper
except ImportError as e:
    print(f"WARNING: Failed to import wrapper modules: {e}")
    feature_extraction_wrapper = None
    registration_wrapper = None 
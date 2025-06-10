"""
Feature Extraction C++ Wrapper

특징점 추출 및 변환을 위한 C++ 래퍼
C++ 구현이 사용 가능할 때는 C++ 함수를 호출하고,
그렇지 않으면 Python 구현으로 Fallback하는 래퍼
"""

import numpy as np
from typing import Union
import warnings

# C++ 모듈 임포트 시도
try:
    from higgsr_ros.core import higgsr_core_cpp
    CPP_AVAILABLE = True
except ImportError:
    higgsr_core_cpp = None
    CPP_AVAILABLE = False

# Python Fallback 구현 임포트
from higgsr_ros.core.feature_extraction import (
    extract_high_density_keypoints as extract_high_density_keypoints_python,
    apply_transform_to_keypoints_numba as apply_transform_to_keypoints_python
)


def _validate_density_extraction_inputs(
    density_map: np.ndarray, 
    x_edges: np.ndarray, 
    y_edges: np.ndarray, 
    density_threshold: float
) -> None:
    """
    밀도 추출 입력 파라미터 유효성 검증
    
    Args:
        density_map: 2D 밀도 맵
        x_edges: X 방향 그리드 경계값
        y_edges: Y 방향 그리드 경계값  
        density_threshold: 밀도 임계값
        
    Raises:
        TypeError: 타입이 올바르지 않은 경우
        ValueError: 값이 유효하지 않은 경우
    """
    if not isinstance(density_map, np.ndarray):
        raise TypeError("density_map must be numpy.ndarray")
    if not isinstance(x_edges, np.ndarray):
        raise TypeError("x_edges must be numpy.ndarray")
    if not isinstance(y_edges, np.ndarray):
        raise TypeError("y_edges must be numpy.ndarray")
    if not isinstance(density_threshold, (int, float)):
        raise TypeError("density_threshold must be numeric")
    
    if density_map.ndim != 2:
        raise ValueError("density_map must be 2-dimensional")
    if x_edges.ndim != 1:
        raise ValueError("x_edges must be 1-dimensional")
    if y_edges.ndim != 1:
        raise ValueError("y_edges must be 1-dimensional")
    
    if x_edges.size < 2:
        raise ValueError("x_edges must have at least 2 elements")
    if y_edges.size < 2:
        raise ValueError("y_edges must have at least 2 elements")
    
    # 그리드 크기 일치성 확인
    if density_map.shape[0] != x_edges.size - 1:
        raise ValueError("density_map.shape[0] must equal x_edges.size - 1")
    if density_map.shape[1] != y_edges.size - 1:
        raise ValueError("density_map.shape[1] must equal y_edges.size - 1")
    
    # 경계값 정렬 확인
    if not np.all(np.diff(x_edges) > 0):
        raise ValueError("x_edges must be in ascending order")
    if not np.all(np.diff(y_edges) > 0):
        raise ValueError("y_edges must be in ascending order")
    
    # NaN/Inf 검증
    if not np.all(np.isfinite(density_map)):
        raise ValueError("density_map contains non-finite values")
    if not np.all(np.isfinite(x_edges)):
        raise ValueError("x_edges contains non-finite values")
    if not np.all(np.isfinite(y_edges)):
        raise ValueError("y_edges contains non-finite values")
    if not np.isfinite(density_threshold):
        raise ValueError("density_threshold must be finite")


def _validate_transform_inputs(
    keypoints_np: np.ndarray,
    tx: float,
    ty: float, 
    theta_rad: float
) -> None:
    """
    변환 입력 파라미터 유효성 검증
    
    Args:
        keypoints_np: 키포인트 배열 (N x 2)
        tx: X 방향 이동
        ty: Y 방향 이동
        theta_rad: 회전 각도(라디안)
        
    Raises:
        TypeError: 타입이 올바르지 않은 경우
        ValueError: 값이 유효하지 않은 경우
    """
    if not isinstance(keypoints_np, np.ndarray):
        raise TypeError("keypoints_np must be numpy.ndarray")
    if not isinstance(tx, (int, float)):
        raise TypeError("tx must be numeric")
    if not isinstance(ty, (int, float)):
        raise TypeError("ty must be numeric")
    if not isinstance(theta_rad, (int, float)):
        raise TypeError("theta_rad must be numeric")
    
    if keypoints_np.ndim != 2 or keypoints_np.shape[1] != 2:
        raise ValueError("keypoints_np must be N x 2 array")
    
    # NaN/Inf 검증
    if not np.all(np.isfinite(keypoints_np)):
        raise ValueError("keypoints_np contains non-finite values")
    if not np.isfinite(tx):
        raise ValueError("tx must be finite")
    if not np.isfinite(ty):
        raise ValueError("ty must be finite")
    if not np.isfinite(theta_rad):
        raise ValueError("theta_rad must be finite")


def extract_high_density_keypoints_cpp(
    density_map: np.ndarray,
    x_edges: np.ndarray,
    y_edges: np.ndarray,
    density_threshold: float,
    use_cpp: bool = True
) -> np.ndarray:
    """
    C++ 가속화된 고밀도 키포인트 추출 함수
    
    Args:
        density_map: 2D 밀도 맵
        x_edges: X 방향 그리드 경계값
        y_edges: Y 방향 그리드 경계값
        density_threshold: 키포인트로 추출할 최소 밀도 임계값
        use_cpp: C++ 구현 사용 여부
        
    Returns:
        numpy.ndarray: 추출된 키포인트 배열 (N x 2)
        
    Raises:
        TypeError: 입력 타입이 올바르지 않은 경우
        ValueError: 입력 값이 유효하지 않은 경우
        RuntimeError: 추출 중 오류가 발생한 경우
    """
    # 입력 유효성 검증
    _validate_density_extraction_inputs(density_map, x_edges, y_edges, density_threshold)
    
    # C++ 구현 시도
    if use_cpp and CPP_AVAILABLE and higgsr_core_cpp:
        try:
            result = higgsr_core_cpp.extract_high_density_keypoints(
                density_map, x_edges, y_edges, density_threshold
            )
            return np.array(result)
        except Exception as e:
            # 🚨 강력한 경고! 사용자가 C++을 기대했지만 Python으로 fallback됨
            print("=" * 70)
            print("🚨 CRITICAL WARNING: C++ FALLBACK TO PYTHON 🚨")
            print("=" * 70)
            print(f"❌ C++ extract_high_density_keypoints FAILED!")
            print(f"📝 Error: {e}")
            print(f"🔄 FALLING BACK TO PYTHON")
            print("=" * 70)
            
            warnings.warn(f"🚨 C++ FAILED, FALLBACK TO PYTHON: {e}", UserWarning, stacklevel=2)
    
    # 🐍 Python Fallback
    return extract_high_density_keypoints_python(density_map, x_edges, y_edges, density_threshold)


def apply_transform_to_keypoints_cpp(
    keypoints_np: np.ndarray,
    tx: float,
    ty: float,
    theta_rad: float,
    use_cpp: bool = True
) -> np.ndarray:
    """
    C++ 가속화된 키포인트 변환 함수
    
    Args:
        keypoints_np: 변환할 키포인트 배열 (N x 2)
        tx: X 방향 이동
        ty: Y 방향 이동
        theta_rad: 회전 각도(라디안)
        use_cpp: C++ 구현 사용 여부
        
    Returns:
        numpy.ndarray: 변환된 키포인트 배열 (N x 2)
        
    Raises:
        TypeError: 입력 타입이 올바르지 않은 경우
        ValueError: 입력 값이 유효하지 않은 경우
        RuntimeError: 변환 중 오류가 발생한 경우
    """
    # 입력 유효성 검증
    _validate_transform_inputs(keypoints_np, tx, ty, theta_rad)
    
    # C++ 구현 시도
    if use_cpp and CPP_AVAILABLE and higgsr_core_cpp:
        try:
            result = higgsr_core_cpp.apply_transform_to_keypoints(
                keypoints_np, tx, ty, theta_rad
            )
            return np.array(result)
        except Exception as e:
            # 🚨 강력한 경고!
            print("=" * 70)
            print("🚨 CRITICAL WARNING: C++ FALLBACK TO PYTHON 🚨")
            print("=" * 70)
            print(f"❌ C++ apply_transform_to_keypoints FAILED!")
            print(f"📝 Error: {e}")
            print(f"🔄 FALLING BACK TO PYTHON")
            print("=" * 70)
            
            warnings.warn(f"🚨 C++ FAILED, FALLBACK TO PYTHON: {e}", UserWarning, stacklevel=2)
    
    # 🐍 Python Fallback
    return apply_transform_to_keypoints_python(keypoints_np, tx, ty, theta_rad)


# 기존 인터페이스와의 호환성을 위한 래퍼 함수들
def extract_high_density_keypoints(density_map, x_edges, y_edges, density_threshold):
    """C++ 우선 실행, 실패 시 Python으로 Fallback (기존 인터페이스 유지)"""
    return extract_high_density_keypoints_cpp(density_map, x_edges, y_edges, density_threshold)


def apply_transform_to_keypoints_numba(keypoints_np, tx, ty, theta_rad):
    """C++ 우선 실행, 실패 시 Python으로 Fallback (기존 인터페이스 유지)"""
    return apply_transform_to_keypoints_cpp(keypoints_np, tx, ty, theta_rad) 
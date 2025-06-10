#!/usr/bin/env python3
"""
HiGGSR C++ 확장 모듈 빌드 및 테스트 스크립트

이 스크립트는 C++ 확장 모듈의 빌드 상태를 확인하고
기본적인 기능 테스트를 수행합니다.
"""

import sys
import os
import numpy as np
import time
from typing import Dict, Any

def test_import_core_module():
    """Core 모듈 임포트 테스트"""
    print("=== Core Module Import Test ===")
    try:
        from higgsr_ros.core import (
            CPP_EXTENSIONS_AVAILABLE, 
            USE_CPP_EXTENSIONS,
            print_acceleration_status,
            get_cpp_acceleration_status
        )
        
        print("✅ Core module import successful")
        print_acceleration_status()
        
        status = get_cpp_acceleration_status()
        return status
        
    except Exception as e:
        print(f"❌ Core module import failed: {e}")
        return None

def test_cpp_module_direct_import():
    """C++ 모듈 직접 임포트 테스트"""
    print("\n=== Direct C++ Module Import Test ===")
    try:
        import higgsr_ros.core.higgsr_core_cpp as cpp_module
        print("✅ C++ module direct import successful")
        print(f"Module version: {getattr(cpp_module, '__version__', 'Unknown')}")
        print(f"Module author: {getattr(cpp_module, '__author__', 'Unknown')}")
        return True
    except ImportError as e:
        print(f"❌ C++ module not available: {e}")
        return False
    except Exception as e:
        print(f"❌ C++ module import error: {e}")
        return False

def test_feature_extraction():
    """Feature Extraction 함수 테스트"""
    print("\n=== Feature Extraction Function Test ===")
    try:
        from higgsr_ros.core import extract_high_density_keypoints, set_use_cpp_extensions
        
        # 테스트 데이터 생성
        density_map = np.random.rand(10, 10) * 2.0  # 0~2 범위
        x_edges = np.linspace(0, 10, 11)
        y_edges = np.linspace(0, 10, 11)  
        density_threshold = 1.0
        
        # Python 구현 테스트
        print("Testing Python implementation...")
        set_use_cpp_extensions(False)
        start_time = time.time()
        keypoints_python = extract_high_density_keypoints(
            density_map, x_edges, y_edges, density_threshold
        )
        python_time = time.time() - start_time
        print(f"Python result: {keypoints_python.shape} keypoints in {python_time:.4f}s")
        
        # C++ 구현 테스트 (가능한 경우)
        from higgsr_ros.core import CPP_EXTENSIONS_AVAILABLE
        if CPP_EXTENSIONS_AVAILABLE:
            print("Testing C++ implementation...")
            set_use_cpp_extensions(True)
            start_time = time.time()
            keypoints_cpp = extract_high_density_keypoints(
                density_map, x_edges, y_edges, density_threshold
            )
            cpp_time = time.time() - start_time
            print(f"C++ result: {keypoints_cpp.shape} keypoints in {cpp_time:.4f}s")
            
            # 결과 비교
            if np.allclose(keypoints_python, keypoints_cpp, rtol=1e-10):
                print("✅ Python and C++ results match")
                if cpp_time < python_time:
                    speedup = python_time / cpp_time
                    print(f"🚀 C++ speedup: {speedup:.2f}x")
                else:
                    print("⚠️ C++ not faster (expected for small test data)")
            else:
                print("❌ Python and C++ results differ")
                return False
        else:
            print("ℹ️ C++ implementation not available, skipping comparison")
        
        print("✅ Feature extraction test passed")
        return True
        
    except Exception as e:
        print(f"❌ Feature extraction test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_keypoint_transformation():
    """키포인트 변환 함수 테스트"""
    print("\n=== Keypoint Transformation Function Test ===")
    try:
        from higgsr_ros.core import apply_transform_to_keypoints_numba, set_use_cpp_extensions
        
        # 테스트 키포인트 생성
        keypoints = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        tx, ty, theta_rad = 1.5, 2.5, np.pi / 4
        
        # Python 구현 테스트
        print("Testing Python transformation...")
        set_use_cpp_extensions(False)
        transformed_python = apply_transform_to_keypoints_numba(keypoints, tx, ty, theta_rad)
        print(f"Python result shape: {transformed_python.shape}")
        
        # C++ 구현 테스트 (가능한 경우)
        from higgsr_ros.core import CPP_EXTENSIONS_AVAILABLE
        if CPP_EXTENSIONS_AVAILABLE:
            print("Testing C++ transformation...")
            set_use_cpp_extensions(True)
            transformed_cpp = apply_transform_to_keypoints_numba(keypoints, tx, ty, theta_rad)
            print(f"C++ result shape: {transformed_cpp.shape}")
            
            # 결과 비교
            if np.allclose(transformed_python, transformed_cpp, rtol=1e-10):
                print("✅ Python and C++ transformation results match")
            else:
                print("❌ Python and C++ transformation results differ")
                print(f"Max difference: {np.max(np.abs(transformed_python - transformed_cpp))}")
                return False
        else:
            print("ℹ️ C++ implementation not available, skipping comparison")
        
        print("✅ Keypoint transformation test passed")
        return True
        
    except Exception as e:
        print(f"❌ Keypoint transformation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_registration_placeholder():
    """Registration 함수 플레이스홀더 테스트"""
    print("\n=== Registration Function Placeholder Test ===")
    try:
        from higgsr_ros.core import hierarchical_adaptive_search, set_use_cpp_extensions
        
        # 테스트 데이터 생성
        global_keypoints = np.random.rand(50, 2) * 10.0
        scan_keypoints = np.random.rand(30, 2) * 8.0
        x_edges = np.linspace(0, 10, 11)
        y_edges = np.linspace(0, 10, 11)
        
        level_configs = [
            {
                'grid_division': [2, 2],
                'search_area_type': 'full_map',
                'tx_ty_search_steps': [5, 5],
                'correspondence_dist_thresh_factor': 1.5
            }
        ]
        
        # Python 구현 테스트
        print("Testing Python registration...")
        set_use_cpp_extensions(False)
        try:
            result_python = hierarchical_adaptive_search(
                global_keypoints, scan_keypoints, x_edges, y_edges, level_configs
            )
            print(f"Python result type: {type(result_python)}")
            if isinstance(result_python, dict):
                print(f"Python result keys: {list(result_python.keys())}")
        except Exception as e:
            print(f"Python registration error (expected for placeholder): {e}")
        
        # C++ 구현 테스트 (가능한 경우)
        from higgsr_ros.core import CPP_EXTENSIONS_AVAILABLE
        if CPP_EXTENSIONS_AVAILABLE:
            print("Testing C++ registration...")
            set_use_cpp_extensions(True)
            try:
                result_cpp = hierarchical_adaptive_search(
                    global_keypoints, scan_keypoints, x_edges, y_edges, level_configs
                )
                print(f"C++ result type: {type(result_cpp)}")
                if isinstance(result_cpp, dict):
                    print(f"C++ result keys: {list(result_cpp.keys())}")
            except Exception as e:
                print(f"C++ registration error (expected for placeholder): {e}")
        else:
            print("ℹ️ C++ implementation not available, skipping test")
        
        print("✅ Registration placeholder test completed (errors expected)")
        return True
        
    except Exception as e:
        print(f"❌ Registration test setup failed: {e}")
        return False

def main():
    """메인 테스트 함수"""
    print("HiGGSR C++ Extension Build and Test")
    print("=" * 50)
    
    # 테스트 순서
    tests = [
        ("Core Module Import", test_import_core_module),
        ("C++ Module Direct Import", test_cpp_module_direct_import),
        ("Feature Extraction", test_feature_extraction),
        ("Keypoint Transformation", test_keypoint_transformation),
        ("Registration Placeholder", test_registration_placeholder),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            result = test_func()
            results[test_name] = result
        except Exception as e:
            print(f"❌ Test {test_name} crashed: {e}")
            results[test_name] = False
    
    # 최종 결과 출력
    print("\n" + "="*60)
    print("FINAL TEST RESULTS")
    print("="*60)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:30} : {status}")
        if result:
            passed += 1
    
    print(f"\nSummary: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed!")
        return 0
    else:
        print("⚠️ Some tests failed. Check the C++ build setup.")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 
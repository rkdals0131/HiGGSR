import numpy as np
from numba import jit
import math

def extract_high_density_keypoints(density_map, x_edges, y_edges, density_threshold):
    """
    밀도 맵에서 임계값보다 높은 밀도를 가진 셀의 중심을 키포인트로 추출합니다.
    
    Args:
        density_map (numpy.ndarray): 2D 밀도 맵
        x_edges (numpy.ndarray): x 방향 그리드 경계
        y_edges (numpy.ndarray): y 방향 그리드 경계
        density_threshold (float): 키포인트로 추출할 최소 밀도 임계값
    
    Returns:
        numpy.ndarray: 추출된 키포인트 배열 (Nx2)
    """
    if density_map.size == 0: 
        return np.array([])
    
    cell_centers_x = (x_edges[:-1] + x_edges[1:]) / 2
    cell_centers_y = (y_edges[:-1] + y_edges[1:]) / 2
    
    keypoints = []
    for i in range(density_map.shape[0]):
        for j in range(density_map.shape[1]):
            if density_map[i, j] > density_threshold:
                keypoints.append([cell_centers_x[i], cell_centers_y[j]])
    
    return np.array(keypoints)

@jit(nopython=True, cache=True)
def apply_transform_to_keypoints_numba(keypoints_np, tx, ty, theta_rad):
    """
    키포인트에 2D 변환(이동 및 회전)을 적용합니다.
    
    Args:
        keypoints_np (numpy.ndarray): 변환할 키포인트 배열 (Nx2)
        tx (float): x 방향 이동
        ty (float): y 방향 이동
        theta_rad (float): 회전 각도(라디안)
    
    Returns:
        numpy.ndarray: 변환된 키포인트 배열 (Nx2)
    """
    if keypoints_np.shape[0] == 0:
        return np.empty((0, 2), dtype=keypoints_np.dtype)
    
    cos_t = math.cos(theta_rad)
    sin_t = math.sin(theta_rad)
    
    transformed_keypoints = np.empty_like(keypoints_np)
    for i in range(keypoints_np.shape[0]):
        x, y = keypoints_np[i, 0], keypoints_np[i, 1]
        transformed_keypoints[i, 0] = x * cos_t - y * sin_t + tx
        transformed_keypoints[i, 1] = x * sin_t + y * cos_t + ty
    
    return transformed_keypoints 
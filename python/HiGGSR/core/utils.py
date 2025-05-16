import numpy as np
import open3d as o3d
from scipy.stats import binned_statistic_2d
import math

def load_point_cloud_from_file(filepath):
    """
    파일에서 포인트 클라우드를 로드합니다.
    
    Args:
        filepath (str): 포인트 클라우드 파일 경로
    
    Returns:
        numpy.ndarray: 로드된 포인트 클라우드 배열 (Nx3)
    """
    try:
        pcd = o3d.io.read_point_cloud(filepath)
        if not pcd.has_points():
            print(f"경고: {filepath}에 포인트가 없습니다.")
            return np.array([])
        return np.asarray(pcd.points)
    except Exception as e:
        print(f"경고: {filepath} 로드 실패 - {str(e)}")
        return np.array([])

def create_2d_height_variance_map(points_3d, grid_cell_size, min_points_per_cell=1, density_metric='std'):
    """
    3D 포인트 클라우드에서 2.5D 높이 분산 맵을 생성합니다.
    
    Args:
        points_3d (numpy.ndarray): 3D 포인트 클라우드 배열 (Nx3)
        grid_cell_size (float): 그리드 셀 크기
        min_points_per_cell (int): 밀도 계산에 필요한 최소 포인트 수
        density_metric (str): 밀도 계산 방법 ('std' 또는 'range')
    
    Returns:
        tuple: (밀도 맵, x 경계, y 경계)
    """
    if points_3d.shape[0] == 0:
        return np.array([]), np.array([]), np.array([])
    
    x_coords = points_3d[:, 0]
    y_coords = points_3d[:, 1]
    z_coords = points_3d[:, 2]
    
    x_min, x_max = np.min(x_coords), np.max(x_coords)
    y_min, y_max = np.min(y_coords), np.max(y_coords)
    
    # 맵이 너무 작을 경우 경계 확장
    if x_max == x_min: 
        x_max = x_min + grid_cell_size
    if y_max == y_min: 
        y_max = y_min + grid_cell_size
    
    # 그리드 경계 계산
    num_bins_x = max(1, int(np.ceil((x_max - x_min) / grid_cell_size)))
    num_bins_y = max(1, int(np.ceil((y_max - y_min) / grid_cell_size)))
    x_edges = np.linspace(x_min, x_min + num_bins_x * grid_cell_size, num_bins_x + 1)
    y_edges = np.linspace(y_min, y_min + num_bins_y * grid_cell_size, num_bins_y + 1)
    
    # 셀당 포인트 수 계산
    count_map, _, _ = np.histogram2d(x_coords, y_coords, bins=[x_edges, y_edges])
    density_map = np.zeros((num_bins_x, num_bins_y))

    # 선택된 메트릭에 따라 밀도 계산
    if density_metric == 'std':
        std_map, _, _, _ = binned_statistic_2d(x_coords, y_coords, z_coords, statistic=np.std, bins=[x_edges, y_edges])
        std_map = np.nan_to_num(std_map)
        density_map = np.where(count_map >= min_points_per_cell, std_map, 0)
    elif density_metric == 'range':
        z_max_map, _, _, _ = binned_statistic_2d(x_coords, y_coords, z_coords, statistic='max', bins=[x_edges, y_edges])
        z_min_map, _, _, _ = binned_statistic_2d(x_coords, y_coords, z_coords, statistic='min', bins=[x_edges, y_edges])
        z_max_map = np.nan_to_num(z_max_map)
        z_min_map = np.nan_to_num(z_min_map)
        temp_density_map = z_max_map - z_min_map
        density_map = np.where(count_map >= min_points_per_cell, temp_density_map, 0)
    else:
        raise ValueError("density_metric은 'std' 또는 'range' 여야 합니다.")
        
    return density_map, x_edges, y_edges

def create_transform_matrix_4x4(tx, ty, theta_deg):
    """
    2D 변환(이동 및 회전)을 위한 4x4 변환 행렬을 생성합니다.
    
    Args:
        tx (float): x 방향 이동
        ty (float): y 방향 이동
        theta_deg (float): 회전 각도(도)
    
    Returns:
        numpy.ndarray: 4x4 변환 행렬
    """
    theta_rad = np.deg2rad(theta_deg)
    cos_t = np.cos(theta_rad)
    sin_t = np.sin(theta_rad)
    
    return np.array([
        [cos_t, -sin_t, 0, tx], 
        [sin_t,  cos_t, 0, ty], 
        [0,      0,     1, 0], 
        [0,      0,     0, 1]
    ]) 
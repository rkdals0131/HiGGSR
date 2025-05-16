import numpy as np
from numba import jit
import math

def extract_high_density_keypoints(density_map, x_edges, y_edges, density_threshold):
    keypoints = []
    if density_map.size == 0: return np.array(keypoints)
    cell_centers_x = (x_edges[:-1] + x_edges[1:]) / 2
    cell_centers_y = (y_edges[:-1] + y_edges[1:]) / 2
    for i in range(density_map.shape[0]):
        for j in range(density_map.shape[1]):
            if density_map[i, j] > density_threshold:
                keypoints.append([cell_centers_x[i], cell_centers_y[j]])
    return np.array(keypoints)

@jit(nopython=True, cache=True)
def apply_transform_to_keypoints_numba(keypoints_np, tx, ty, theta_rad):
    if keypoints_np.shape[0] == 0:
        return np.empty((0, 2), dtype=keypoints_np.dtype)
    cos_t = math.cos(theta_rad); sin_t = math.sin(theta_rad)
    transformed_keypoints = np.empty_like(keypoints_np)
    for i in range(keypoints_np.shape[0]):
        x = keypoints_np[i, 0]; y = keypoints_np[i, 1]
        transformed_keypoints[i, 0] = x * cos_t - y * sin_t + tx
        transformed_keypoints[i, 1] = x * sin_t + y * cos_t + ty
    return transformed_keypoints 
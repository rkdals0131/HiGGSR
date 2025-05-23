import numpy as np
from sensor_msgs_py import point_cloud2

def convert_ros_point_cloud2_to_numpy(cloud_msg, field_names=('x', 'y', 'z')):
    """
    sensor_msgs/PointCloud2 메시지를 NumPy 배열로 변환합니다.

    Args:
        cloud_msg (sensor_msgs.msg.PointCloud2): 입력 포인트 클라우드 메시지.
        field_names (tuple, optional): 추출할 필드 이름. 기본값은 ('x', 'y', 'z').

    Returns:
        numpy.ndarray: NxM 형태의 NumPy 배열 (N: 포인트 수, M: 필드 수).
                       필드가 존재하지 않거나 데이터가 없으면 None을 반환.
    """
    try:
        # PointCloud2 메시지에서 특정 필드의 데이터를 NumPy 배열로 읽어옵니다.
        gen = point_cloud2.read_points(cloud_msg, field_names=field_names, skip_nans=True)
        points_list = list(gen)
        if not points_list:
            return None
        
        # structured array 문제 해결: 각 필드를 개별적으로 추출
        if isinstance(points_list[0], tuple):
            # 튜플 형태인 경우 (정상적인 경우)
            points_np = np.array(points_list, dtype=np.float32)
        else:
            # structured array인 경우 각 필드를 개별적으로 추출
            num_points = len(points_list)
            num_fields = len(field_names)
            points_np = np.zeros((num_points, num_fields), dtype=np.float32)
            
            for i, point in enumerate(points_list):
                for j, field_name in enumerate(field_names):
                    points_np[i, j] = float(point[field_name])
                    
        return points_np
    except Exception as e:
        print(f"Error converting PointCloud2 to NumPy: {e}")
        return None 
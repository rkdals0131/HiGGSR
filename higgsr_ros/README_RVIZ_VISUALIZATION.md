# HiGGSR RViz2 시각화 시스템

이 가이드는 HiGGSR 시스템의 RViz2 시각화 기능을 사용하는 방법을 설명합니다.

## 개요

HiGGSR RViz2 시각화 시스템은 포인트 클라우드 정합 결과를 실시간으로 시각화할 수 있는 기능을 제공합니다. 다음과 같은 시각화 요소들을 지원합니다:

- **글로벌 맵**: 흰색 포인트클라우드로 표시
- **라이브 스캔**: 빨간색 포인트클라우드로 표시 (정합 후 변환 적용됨)
- **글로벌 키포인트**: 파란색 구형 마커로 표시
- **스캔 키포인트**: 녹색 구형 마커로 표시 (정합 후 변환 적용됨)
- **로봇 포즈**: 화살표 마커로 현재 위치 및 방향 표시
- **경로**: 로봇의 이동 경로를 선으로 표시
- **통계 정보**: 텍스트로 거리 및 포즈 수 표시

## 시스템 요구사항

- ROS2 Humble
- RViz2
- Python 3.10+
- 필요한 Python 패키지: numpy, open3d, scipy, numba

## 설치 및 빌드

1. 워크스페이스에서 패키지 빌드:
```bash
cd /path/to/your/workspace
colcon build --packages-select higgsr_ros --symlink-install
source install/setup.zsh  # 또는 setup.bash
```

## 사용 방법

### 1. 전체 시각화 시스템 실행

가장 간단한 방법은 launch 파일을 사용하는 것입니다:

```bash
ros2 launch higgsr_ros higgsr_visualization_launch.py
```

이 명령은 다음을 실행합니다:
- 파일 처리 노드 (`file_processor_node`)
- 시각화 노드 (`higgsr_visualization_node`)
- RViz2 (미리 설정된 설정 파일 사용)
- 정적 TF 브로드캐스터 (map 프레임 생성)

### 2. 개별 노드 실행

필요에 따라 각 노드를 개별적으로 실행할 수도 있습니다:

```bash
# 터미널 1: 파일 처리 노드
ros2 run higgsr_ros file_processor_node

# 터미널 2: 시각화 노드
ros2 run higgsr_ros higgsr_visualization_node

# 터미널 3: RViz2
ros2 run rviz2 rviz2 -d src/higgsr_ros/config/higgsr_visualization.rviz

# 터미널 4: 정적 TF (필요시)
ros2 run tf2_ros static_transform_publisher 0 0 0 0 0 0 world map
```

### 3. 테스트 실행

시스템이 정상적으로 실행된 후, 새 터미널에서 테스트를 실행합니다:

```bash
ros2 run higgsr_ros test_rviz_visualization
```

이 테스트는:
1. 기본 데이터 파일을 사용하여 정합을 수행합니다
2. 결과를 RViz2로 퍼블리시합니다
3. 시각화 결과를 확인할 수 있도록 노드를 유지합니다

## RViz2 설정

### 프레임 설정
- **Fixed Frame**: `map`
- 모든 시각화 요소들은 `map` 프레임을 기준으로 표시됩니다

### 표시 요소들

1. **Grid**: 기본 격자 (회색)
2. **HiGGSR Path**: 로봇 경로 (녹색 선)
3. **HiGGSR Pose**: 현재 로봇 포즈 (빨간색 화살표)
4. **HiGGSR Trajectory**: 궤적 (파란색 선)
5. **Global Map PointCloud**: 글로벌 맵 (흰색 점들)
6. **Live Scan PointCloud**: 변환된 스캔 데이터 (빨간색 점들)
7. **Global Keypoints**: 글로벌 키포인트 (파란색 구)
8. **Scan Keypoints**: 변환된 스캔 키포인트 (녹색 구)
9. **HiGGSR Markers**: 통계 정보 텍스트

### 뷰 설정
- **Camera Type**: Orbit
- **Distance**: 50m
- **Focal Point**: (0, 0, 0)

## 토픽 구조

### 퍼블리시되는 토픽들

| 토픽 이름 | 메시지 타입 | 설명 |
|-----------|------------|------|
| `/higgsr_global_map` | `sensor_msgs/PointCloud2` | 글로벌 맵 포인트클라우드 |
| `/higgsr_live_scan` | `sensor_msgs/PointCloud2` | 변환된 라이브 스캔 |
| `/higgsr_global_keypoints` | `visualization_msgs/MarkerArray` | 글로벌 키포인트 마커 |
| `/higgsr_scan_keypoints` | `visualization_msgs/MarkerArray` | 변환된 스캔 키포인트 마커 |
| `/higgsr_pose` | `geometry_msgs/PoseStamped` | 추정된 로봇 포즈 |
| `/higgsr_pose_marker` | `visualization_msgs/Marker` | 포즈 화살표 마커 |
| `/higgsr_path` | `nav_msgs/Path` | 로봇 경로 |
| `/higgsr_trajectory` | `visualization_msgs/Marker` | 궤적 선 마커 |
| `/higgsr_markers` | `visualization_msgs/MarkerArray` | 통계 정보 마커 |

### 서비스

| 서비스 이름 | 서비스 타입 | 설명 |
|------------|-------------|------|
| `/process_files` | `higgsr_interface/ProcessFiles` | 파일 기반 정합 처리 |

## 파라미터 설정

### 파일 처리 노드 파라미터

```yaml
# 알고리즘 파라미터
grid_size: 0.2
min_points_for_density_calc: 3
density_metric: 'std'  # 'std' or 'range'
keypoint_density_threshold: 0.1
num_processes: 0  # 0이면 자동 설정

# 프레임 설정
global_frame_id: 'map'
scan_frame_id: 'base_link'

# 시각화 활성화 설정
enable_pillar_maps_visualization: false
enable_2d_keypoints_visualization: true
enable_super_grid_heatmap_visualization: false
enable_3d_result_visualization: true
```

### 시각화 노드 파라미터

```yaml
# 프레임 설정
map_frame_id: 'map'
base_frame_id: 'base_link'

# 시각화 설정
visualization_scale: 1.0
path_max_length: 100
marker_lifetime: 30.0  # 초
```

## 트러블슈팅

### 1. RViz2가 실행되지 않는 경우
```bash
# RViz2 패키지 설치 확인
sudo apt install ros-humble-rviz2

# 환경 변수 확인
echo $ROS_DISTRO
source /opt/ros/humble/setup.zsh
```

### 2. 토픽이 표시되지 않는 경우
```bash
# 토픽 목록 확인
ros2 topic list | grep higgsr

# 특정 토픽 데이터 확인
ros2 topic echo /higgsr_global_map --once
```

### 3. 프레임 오류가 발생하는 경우
```bash
# TF 트리 확인
ros2 run tf2_tools view_frames.py

# 정적 TF 수동 실행
ros2 run tf2_ros static_transform_publisher 0 0 0 0 0 0 world map
```

### 4. 노드가 연결되지 않는 경우
```bash
# 노드 목록 확인
ros2 node list

# 서비스 목록 확인
ros2 service list | grep process_files
```

## 예제 실행 순서

완전한 테스트를 위한 단계별 실행 방법:

1. **환경 준비**:
```bash
cd /path/to/your/workspace
source install/setup.zsh
```

2. **시스템 실행**:
```bash
# 터미널 1
ros2 launch higgsr_ros higgsr_visualization_launch.py
```

3. **테스트 실행**:
```bash
# 터미널 2
ros2 run higgsr_ros test_rviz_visualization
```

4. **결과 확인**:
   - RViz2 창에서 시각화 결과 확인
   - 포인트클라우드와 키포인트가 올바르게 정렬되었는지 확인
   - 터미널에서 정합 결과 로그 확인

## 사용자 정의

### 새로운 데이터 파일 사용

테스트 스크립트를 수정하여 다른 파일을 사용할 수 있습니다:

```python
# test_rviz_visualization.py 에서
success = node.test_visualization(
    global_map_path="/path/to/your/global_map.ply",
    live_scan_path="/path/to/your/live_scan.ply"
)
```

### RViz2 설정 수정

`config/higgsr_visualization.rviz` 파일을 편집하여 시각화 설정을 변경할 수 있습니다.

## 성능 최적화

- **포인트클라우드 크기**: 큰 포인트클라우드는 시각화 성능에 영향을 줄 수 있습니다
- **키포인트 수**: 키포인트가 너무 많으면 마커 렌더링이 느려질 수 있습니다
- **업데이트 주기**: 필요에 따라 시각화 업데이트 주기를 조정할 수 있습니다

```python
# visualization_node.py에서 타이머 주기 변경
self.visualization_timer = self.create_timer(0.1, self.update_visualization)  # 0.1초
```

## 추가 정보

더 자세한 정보는 다음 파일들을 참고하세요:
- `README.md`: 전체 시스템 개요
- `README_FILE_PROCESSOR.md`: 파일 처리 기능 상세 설명
- `higgsr_ros/visualization/visualization_node.py`: 시각화 노드 구현
- `higgsr_ros/nodes/file_processor_node.py`: 파일 처리 노드 구현 
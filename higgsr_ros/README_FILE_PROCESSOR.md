# HiGGSR 파일 처리 기능 사용 가이드

본 기능은 `main.py`와 동일한 파일 기반 처리를 ROS 환경에서 제공합니다.

## 개요

파일 처리 기능을 통해 `.ply` 파일로 저장된 전역 맵과 라이브 스캔 데이터를 직접 처리하여 위치 정합을 수행할 수 있습니다. 모든 알고리즘 파라미터를 설정 파일이나 JSON을 통해 제어할 수 있습니다.

## 구성 요소

### 1. 서비스 정의
- **ProcessFiles.srv**: 파일 경로와 설정을 입력받아 정합 결과를 반환하는 서비스

### 2. 노드들
- **file_processor_node**: 파일 처리 서비스를 제공하는 서버 노드
- **file_processor_client_node**: 파일 처리 서비스를 호출하는 클라이언트 노드

### 3. 설정 파일
- **file_processor_config.yaml**: 알고리즘 파라미터 설정 파일

## 사용 방법

### 1. 기본 사용 (서버-클라이언트 분리)

#### 서버 실행
```bash
# 기본 설정으로 실행
ros2 launch higgsr_ros file_processor.launch.py

# 커스텀 설정 파일로 실행
ros2 launch higgsr_ros file_processor.launch.py config_file:=/path/to/custom_config.yaml
```

#### 클라이언트로 요청
```bash
# 기본 요청
ros2 run higgsr_ros file_processor_client_node \
  --global-map /path/to/global_map.ply \
  --live-scan /path/to/live_scan.ply

# 빠른 처리 모드
ros2 run higgsr_ros file_processor_client_node \
  --global-map /path/to/global_map.ply \
  --live-scan /path/to/live_scan.ply \
  --quick

# 정확한 처리 모드
ros2 run higgsr_ros file_processor_client_node \
  --global-map /path/to/global_map.ply \
  --live-scan /path/to/live_scan.ply \
  --accurate

# 커스텀 JSON 설정
ros2 run higgsr_ros file_processor_client_node \
  --global-map /path/to/global_map.ply \
  --live-scan /path/to/live_scan.ply \
  --config-json '{"grid_size": 0.3, "keypoint_density_threshold": 0.12}'

# 커스텀 JSON 파일
ros2 run higgsr_ros file_processor_client_node \
  --global-map /path/to/global_map.ply \
  --live-scan /path/to/live_scan.ply \
  --config-file /path/to/config.json
```

### 2. 단독 노드 실행 (서비스 없이)

서비스 없이 바로 처리하고 싶다면 file_processor_node를 직접 실행하고 별도 스크립트로 서비스를 호출할 수 있습니다.

## 설정 파라미터

### 알고리즘 파라미터

| 파라미터 | 기본값 | 설명 |
|---------|--------|------|
| `grid_size` | 0.2 | 그리드 셀 크기 (미터) |
| `min_points_for_density_calc` | 3 | 밀도 계산에 필요한 최소 포인트 수 |
| `density_metric` | 'std' | 밀도 메트릭 ('std' 또는 'range') |
| `keypoint_density_threshold` | 0.1 | 키포인트 밀도 임계값 |
| `num_processes` | 0 | 병렬 처리 프로세스 수 (0이면 자동) |
| `num_candidates_per_level` | 3 | 레벨당 후보 수 |
| `min_candidate_separation_factor` | 1.5 | 최소 후보 분리 계수 |

### 계층적 탐색 설정 (level_configs)

JSON 형태로 각 레벨의 탐색 전략을 설정할 수 있습니다:

```json
[
  {
    "grid_division": [6, 6],
    "search_area_type": "full_map",
    "theta_range_deg": [0, 359],
    "theta_search_steps": 48,
    "correspondence_distance_threshold_factor": 7.0,
    "tx_ty_search_steps_per_cell": [10, 10]
  }
]
```

### 시각화 설정

| 파라미터 | 기본값 | 설명 |
|---------|--------|------|
| `enable_pillar_maps_visualization` | false | Pillar Map 시각화 |
| `enable_2d_keypoints_visualization` | true | 2D 키포인트 시각화 |
| `enable_super_grid_heatmap_visualization` | true | 탐색 히트맵 시각화 |
| `enable_3d_result_visualization` | true | 3D 결과 시각화 |

## 예제 설정 파일

### 빠른 처리용 설정
```yaml
file_processor_node:
  ros__parameters:
    grid_size: 0.4
    keypoint_density_threshold: 0.15
    level_configs: >
      [
        {
          "grid_division": [4, 4],
          "search_area_type": "full_map",
          "theta_range_deg": [0, 359], "theta_search_steps": 24,
          "correspondence_distance_threshold_factor": 10.0,
          "tx_ty_search_steps_per_cell": [5, 5]
        }
      ]
    enable_3d_result_visualization: true
    enable_2d_keypoints_visualization: false
    enable_super_grid_heatmap_visualization: false
```

### 정확한 처리용 설정
```yaml
file_processor_node:
  ros__parameters:
    grid_size: 0.1
    keypoint_density_threshold: 0.05
    level_configs: >
      [
        {
          "grid_division": [8, 8],
          "search_area_type": "full_map",
          "theta_range_deg": [0, 359], "theta_search_steps": 72,
          "correspondence_distance_threshold_factor": 5.0,
          "tx_ty_search_steps_per_cell": [15, 15]
        },
        {
          "grid_division": [10, 10],
          "search_area_type": "relative_to_map", "area_ratio_or_size": 0.3,
          "theta_range_deg_relative": [0, 359], "theta_search_steps": 72,
          "correspondence_distance_threshold_factor": 3.0,
          "tx_ty_search_steps_per_cell": [15, 15]
        }
      ]
    enable_pillar_maps_visualization: true
    enable_2d_keypoints_visualization: true
    enable_super_grid_heatmap_visualization: true
    enable_3d_result_visualization: true
```

## 출력 결과

성공적인 처리 시 다음 정보가 반환됩니다:

- **성공 여부**: boolean
- **추정된 변환**: geometry_msgs/TransformStamped
- **정합 점수**: float32
- **처리 시간**: float32 (초)
- **계산 반복 수**: int32
- **메시지**: 상태 메시지

## 활용 예시

### 1. 배치 처리
여러 스캔 파일을 순차적으로 처리하는 스크립트를 작성할 수 있습니다.

### 2. 성능 테스트
다양한 파라미터 조합으로 성능을 테스트할 수 있습니다.

### 3. 오프라인 분석
실시간이 아닌 환경에서 정밀한 분석을 수행할 수 있습니다.

## 문제 해결

### 일반적인 오류들

1. **파일을 찾을 수 없음**: 파일 경로가 정확한지 확인
2. **서비스 연결 실패**: file_processor_node가 실행 중인지 확인
3. **메모리 부족**: num_processes 값을 줄이거나 grid_size를 늘려보세요
4. **정합 실패**: keypoint_density_threshold를 조정해보세요

### 로그 확인
```bash
ros2 node list
ros2 service list
ros2 service type /process_files
```

## 원본 main.py와의 차이점

- ROS 서비스 인터페이스 제공
- 설정 파일 기반 파라미터 관리
- 결과의 ROS 메시지 형태 반환
- 멀티 노드 구조로 확장성 향상 
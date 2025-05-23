# HiGGSR ROS2 시스템

HiGGSR (Hierarchical Global Grid Search and Registration)은 3D LiDAR 스캔 정합을 위한 ROS2 기반 시스템입니다. 계층적 그리드 기반 탐색을 통해 빠르고 정확한 위치 추정을 제공합니다.

## 📋 목차
- [시스템 개요](#-시스템-개요)
- [노드별 기능](#-노드별-기능)
- [설정 파일](#-설정-파일)
- [사용법](#-사용법)
- [토픽 및 서비스](#-토픽-및-서비스)
- [의존성 및 설치](#-의존성-및-설치)

## 🏗️ 시스템 개요

HiGGSR 시스템은 개별 노드 기반으로 구성되어 있어 필요한 기능만 선택적으로 실행할 수 있습니다:

```
HiGGSR ROS2 System (개별 노드 실행 방식)
├── higgsr_server_node              # 핵심 스캔 정합 서버
├── lidar_client_node               # 라이다 데이터 처리 클라이언트
├── higgsr_visualization_node       # 결과 시각화
└── file_processor_node             # 파일 기반 배치 처리
```

## 🔧 노드별 기능

### 1. HiGGSR Server Node (`higgsr_server_node`)
**핵심 알고리즘 처리 서버**

**주요 기능:**
- 시작 시 글로벌 맵 자동 로드 (`HiGGSR/Data/around_singong - Cloud.ply`)
- 글로벌 맵 전처리 (Pillar Map 생성, 키포인트 추출)
- 라이브 스캔 정합 서비스 제공 (`/register_scan`)
- 계층적 그리드 검색 알고리즘 실행

**입력:**
- 라이브 스캔 포인트클라우드 (서비스 요청 시)

**출력:**
- 추정된 변환 행렬 (위치 및 자세)
- 정합 점수
- 처리 시간 정보

**실행 방법:**
```bash
ros2 run higgsr_ros higgsr_server_node --ros-args --params-file config/higgsr_server_config.yaml
```

### 2. LiDAR Client Node (`lidar_client_node`)
**라이다 데이터 처리 클라이언트**

**주요 기능:**
- 라이다 토픽 구독 및 데이터 저장
- 수동 스캔 정합 요청 (키보드 입력 기반)
- 결과를 ROS2 메시지로 퍼블리시
- TF 변환 브로드캐스트

**입력:**
- 라이다 포인트클라우드 토픽 (설정 가능)
- 키보드 입력 (스페이스바/엔터키: 정합 요청, 'q': 종료)

**출력:**
- `/higgsr_pose`: 추정된 위치 (PoseStamped)
- `/higgsr_transform`: 변환 행렬 (TransformStamped)
- TF 변환 (map_higgsr → base_link)

**실행 방법:**
```bash
ros2 run higgsr_ros lidar_client_node --ros-args --params-file config/lidar_client_config.yaml
```

**사용법:**
- 노드 실행 후 터미널에서 스페이스바나 엔터키를 눌러 스캔 정합 요청
- 'q' 입력으로 종료

### 3. Visualization Node (`higgsr_visualization_node`)
**결과 시각화 노드**

**주요 기능:**
- 글로벌 맵 포인트클라우드 시각화
- 라이브 스캔 포인트클라우드 시각화
- 키포인트 마커 표시
- 로봇 경로 추적 및 표시
- 실시간 통계 정보 표시

**입력:**
- `/higgsr_pose`: 로봇 위치 정보
- `/higgsr_transform`: 변환 행렬 정보

**출력:**
- `/higgsr_global_map`: 글로벌 맵 포인트클라우드
- `/higgsr_live_scan`: 라이브 스캔 시각화
- `/higgsr_global_keypoints`: 글로벌 키포인트 마커
- `/higgsr_scan_keypoints`: 스캔 키포인트 마커
- `/higgsr_path`: 로봇 경로
- `/higgsr_trajectory`: 궤적 라인 마커
- `/higgsr_pose_marker`: 현재 위치 마커

**실행 방법:**
```bash
ros2 run higgsr_ros higgsr_visualization_node --ros-args --params-file config/visualization_config.yaml
```

### 4. File Processor Node (`file_processor_node`)
**파일 기반 배치 처리 노드**

**주요 기능:**
- PLY 파일에서 포인트클라우드 로드
- 파일 기반 정합 처리 (실시간이 아닌 배치 작업)
- 처리 결과 저장 및 시각화
- 디버그 정보 및 통계 생성

**특징:**
- 실시간 처리가 아닌 배치 처리
- 다양한 파일 형식 지원
- 처리 결과 로그 저장
- 성능 분석 정보 제공

**실행 방법:**
```bash
ros2 run higgsr_ros file_processor_node --ros-args --params-file config/file_processor_config.yaml
```

## ⚙️ 설정 파일

모든 노드의 파라미터는 YAML 설정 파일로 관리됩니다:

### 1. `config/higgsr_server_config.yaml`
**HiGGSR 서버 노드 설정**
- 글로벌 맵 파일 경로 및 처리 파라미터
- 라이브 스캔 처리 파라미터
- 계층적 검색 알고리즘 설정
- 병렬 처리 및 성능 관련 설정

### 2. `config/lidar_client_config.yaml`
**라이다 클라이언트 노드 설정**
- 라이다 토픽 및 프레임 설정
- TF 퍼블리시 설정
- 결과 토픽 설정

### 3. `config/visualization_config.yaml`
**시각화 노드 설정**
- 마커 스타일 및 색상 설정
- 시각화 업데이트 주기
- 포인트클라우드 및 경로 설정

### 4. `config/file_processor_config.yaml`
**파일 처리 노드 설정**
- 입출력 파일 경로
- 배치 처리 파라미터
- 시각화 및 로깅 설정

## 📖 사용법

### 기본 시스템 실행 (추천)
```bash
# 1. HiGGSR 서버 시작 (첫 번째 터미널)
ros2 run higgsr_ros higgsr_server_node --ros-args --params-file config/higgsr_server_config.yaml

# 2. 라이다 클라이언트 시작 (두 번째 터미널)
ros2 run higgsr_ros lidar_client_node --ros-args --params-file config/lidar_client_config.yaml

# 3. 시각화 노드 시작 (세 번째 터미널, 선택사항)
ros2 run higgsr_ros higgsr_visualization_node --ros-args --params-file config/visualization_config.yaml

# 4. RViz2 실행 (네 번째 터미널, 선택사항)
rviz2 -d config/higgsr_visualization.rviz
```

### 개별 노드 실행
```bash
# 서버만 실행
ros2 run higgsr_ros higgsr_server_node

# 커스텀 라이다 토픽으로 클라이언트 실행
ros2 run higgsr_ros lidar_client_node --ros-args -p lidar_topic:=/velodyne_points

# 시각화만 실행
ros2 run higgsr_ros higgsr_visualization_node

# 파일 처리 실행
ros2 run higgsr_ros file_processor_node
```

### 파라미터 오버라이드
```bash
# 특정 파라미터 변경
ros2 run higgsr_ros higgsr_server_node --ros-args \
    -p global_grid_size:=0.15 \
    -p num_processes:=4

# 여러 파라미터 동시 변경
ros2 run higgsr_ros lidar_client_node --ros-args \
    -p lidar_topic:=/points \
    -p publish_tf:=false \
    -p base_frame_id:=robot_base
```

### 서비스 직접 호출
```bash
# 스캔 정합 서비스 상태 확인
ros2 service list | grep register_scan
ros2 service type /register_scan

# 서비스 직접 테스트 (포인트클라우드 토픽이 있을 때)
ros2 service call /register_scan higgsr_interface/srv/RegisterScan
```

## 🌐 토픽 및 서비스

### 주요 토픽
| 토픽 이름 | 메시지 타입 | 설명 |
|-----------|-------------|------|
| `/points` | `sensor_msgs/PointCloud2` | 입력 라이다 포인트클라우드 |
| `/higgsr_pose` | `geometry_msgs/PoseStamped` | 추정된 로봇 위치 |
| `/higgsr_transform` | `geometry_msgs/TransformStamped` | 변환 행렬 |
| `/higgsr_global_map` | `sensor_msgs/PointCloud2` | 글로벌 맵 시각화 |
| `/higgsr_live_scan` | `sensor_msgs/PointCloud2` | 라이브 스캔 시각화 |
| `/higgsr_path` | `nav_msgs/Path` | 로봇 경로 |
| `/higgsr_global_keypoints` | `visualization_msgs/MarkerArray` | 글로벌 키포인트 마커 |
| `/higgsr_scan_keypoints` | `visualization_msgs/MarkerArray` | 스캔 키포인트 마커 |

### 서비스
| 서비스 이름 | 서비스 타입 | 설명 |
|-------------|-------------|------|
| `/register_scan` | `higgsr_interface/srv/RegisterScan` | 스캔 정합 요청 |
| `/set_global_map` | `higgsr_interface/srv/SetGlobalMap` | 글로벌 맵 설정 (레거시) |

### TF 프레임
- `map_higgsr`: 글로벌 맵 프레임
- `base_link`: 로봇 베이스 프레임
- `odom`: 오도메트리 프레임

## 📦 의존성 및 설치

### ROS2 패키지
- `rclpy`
- `sensor_msgs`
- `geometry_msgs`
- `visualization_msgs`
- `nav_msgs`
- `tf2_ros`
- `higgsr_interface` (커스텀 인터페이스)

### 파이썬 패키지
```bash
pip3 install open3d numpy matplotlib scipy
```

### 시스템 요구사항
- ROS2 Humble 이상
- Python 3.8 이상
- Ubuntu 20.04 이상

### 빌드 및 설치
```bash
# 1. 워크스페이스로 이동
cd /home/user1/ROS2_Workspace/higgsros_ws

# 2. 의존성 설치
rosdep install --from-paths src --ignore-src -r -y

# 3. 패키지 빌드
colcon build --packages-select higgsr_interface higgsr_ros

# 4. 환경 소싱 (zsh 사용자)
source install/setup.zsh

# 5. .zshrc에 자동 소싱 추가
echo "source /home/user1/ROS2_Workspace/higgsros_ws/install/setup.zsh" >> ~/.zshrc
```

## 🔧 트러블슈팅

### 일반적인 문제들

#### 1. 노드를 찾을 수 없음
```bash
# 패키지가 제대로 빌드되고 소싱되었는지 확인
source install/setup.zsh
ros2 pkg list | grep higgsr
```

#### 2. 설정 파일을 찾을 수 없음
```bash
# 상대 경로로 설정 파일 지정
ros2 run higgsr_ros higgsr_server_node --ros-args --params-file src/higgsr_ros/config/higgsr_server_config.yaml
```

#### 3. 글로벌 맵 파일을 찾을 수 없음
```bash
# 파일 경로 확인
ls -la src/HiGGSR/Data/
# 파라미터로 경로 수정
ros2 run higgsr_ros higgsr_server_node --ros-args -p global_map_file_path:="/full/path/to/map.ply"
```

#### 4. 서비스가 응답하지 않음
```bash
# 서비스 상태 확인
ros2 service list
ros2 service type /register_scan
ros2 node info /higgsr_server_node
```

#### 5. TF 오류
```bash
# TF 트리 확인
ros2 run tf2_tools view_frames
ros2 run tf2_ros tf2_echo map_higgsr base_link
```

### 성능 최적화

#### 고속 처리 설정
```bash
ros2 run higgsr_ros higgsr_server_node --ros-args \
    -p live_grid_size:=0.3 \
    -p num_processes:=4 \
    -p num_candidates_per_level:=2
```

#### 고정밀 처리 설정
```bash
ros2 run higgsr_ros higgsr_server_node --ros-args \
    -p global_grid_size:=0.15 \
    -p live_grid_size:=0.15 \
    -p global_keypoint_density_threshold:=0.05
```

## 📈 모니터링

### 시스템 상태 확인
```bash
# 실행 중인 노드 확인
ros2 node list

# 토픽 모니터링
ros2 topic hz /higgsr_pose
ros2 topic echo /higgsr_transform --no-arr

# 서비스 테스트
ros2 service call /register_scan higgsr_interface/srv/RegisterScan
```

### 로그 확인
```bash
# 특정 노드 로그 확인
ros2 log list
ros2 log level /higgsr_server_node DEBUG
```

## 📄 라이센스

MIT License

## 🤝 기여하기

버그 리포트, 기능 요청, 풀 리퀘스트를 환영합니다! 
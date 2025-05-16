# HiGGSR: 계층적 전역 그리드 탐색 및 정합 (Hierarchical Global Grid Search and Registration)

HiGGSR는 3D 포인트 클라우드를 위한 효율적인 전역 정합 알고리즘을 제공하는 파이썬 패키지입니다. 계층적 그리드 기반 탐색 방법을 통해 빠르고 정확한 정합 결과를 찾습니다.

## 주요 특징

- **계층적 검색**: 넓은 탐색 공간을 여러 단계로 분할하여 효율적으로 검색
- **Pillar Maps**: 2.5D 그리드 맵 기반 특징 추출
- **다양한 시각화 도구**: 정합 과정과 결과 시각화
- **병렬 처리 지원**: 멀티프로세싱을 통한 계산 속도 향상

## 설치 방법

```bash
git clone https://github.com/rkdals0131/HiGGSR.git
cd HiGGSR
pip install -e .
```

## 사용 예시

```python
import numpy as np
from HiGGSR import core
from HiGGSR import visualization as viz

# 포인트 클라우드 로드
global_map_points_3d = core.load_point_cloud_from_file("global_map.ply")
live_scan_points_3d = core.load_point_cloud_from_file("live_scan.ply")

# Pillar Map 생성
grid_cell_size = 0.2
min_points_per_cell = 3
density_map_global, x_edges_global, y_edges_global = core.create_2d_height_variance_map(
    global_map_points_3d, grid_cell_size, min_points_per_cell, 'std')

# 키포인트 추출
global_keypoints = core.extract_high_density_keypoints(density_map_global, x_edges_global, y_edges_global, 0.1)

# 설정 및 실행 
# ...
```

## 모듈 구조

- **core**: 핵심 알고리즘과 연산 함수들
- **visualization**: 시각화 관련 함수들

## 라이센스

MIT License 
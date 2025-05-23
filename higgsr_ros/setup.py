from setuptools import setup
import os
from glob import glob

package_name = 'higgsr_ros'
submodules = [
    'higgsr_ros.core',
    'higgsr_ros.utils',
    'higgsr_ros.nodes',
    'higgsr_ros.visualization',
    'higgsr_ros.scripts'
]

setup(
    name=package_name,
    version='0.2.0',
    packages=[package_name] + submodules,
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # 설정 파일들만 포함 (launch 파일 제거)
        (os.path.join('share', package_name, 'config'), glob('config/*.rviz')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    ],
    install_requires=[
        'setuptools', 
        'numpy', 
        'open3d', 
        'scipy',
        'numba'  # HiGGSR 핵심 알고리즘 최적화를 위한 의존성
    ],
    zip_safe=True,
    maintainer='user1',
    maintainer_email='kikiws70@gmail.com',
    description='개별 노드 기반 HiGGSR 로컬라이제이션 시스템 - 필요한 기능만 선택적으로 실행 가능',
    license='Apache License 2.0',
    extras_require={'test': ['pytest']},
    entry_points={
        'console_scripts': [
            # === 메인 시스템 노드들 ===
            'higgsr_server_node = higgsr_ros.nodes.higgsr_server_node:main',            # 핵심 스캔 정합 서버
            'lidar_client_node = higgsr_ros.nodes.lidar_client_node:main',              # 라이다 데이터 처리 클라이언트
            'higgsr_visualization_node = higgsr_ros.visualization.visualization_node:main',  # 결과 시각화
            'file_processor_node = higgsr_ros.nodes.file_processor_node:main',          # 파일 기반 배치 처리
            'file_processor_client_node = higgsr_ros.nodes.file_processor_client_node:main',  # 파일 처리 클라이언트
            
            # === 유틸리티 스크립트들 ===
            'test_higgsr_system = higgsr_ros.scripts.test_higgsr_system:main',          # 시스템 테스트
            'test_rviz_visualization = higgsr_ros.scripts.test_rviz_visualization:main',  # RViz 시각화 테스트
        ],
    },
)

from setuptools import setup, find_packages
import os
from glob import glob

package_name = 'higgsr_ros'

setup(
    name=package_name,
    version='0.2.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # 설정 파일들만 포함
        (os.path.join('share', package_name, 'config'), glob('config/*.rviz')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    ],
    install_requires=[
        'setuptools', 
        'numpy', 
        'open3d', 
        'scipy',
        'numba'
    ],
    zip_safe=True,
    maintainer='user1',
    maintainer_email='kikiws70@gmail.com',
    description='HiGGSR 로컬라이제이션 시스템',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            # 메인 시스템 노드들
            'higgsr_server_node = higgsr_ros.nodes.higgsr_server_node:main',
            'lidar_client_node = higgsr_ros.nodes.lidar_client_node:main',
            'higgsr_visualization_node = higgsr_ros.visualization.visualization_node:main',
            'file_processor_node = higgsr_ros.nodes.file_processor_node:main',
            'file_processor_client_node = higgsr_ros.nodes.file_processor_client_node:main',
            
            # 유틸리티 스크립트들
            'test_higgsr_system = higgsr_ros.scripts.test_higgsr_system:main',
            'test_rviz_visualization = higgsr_ros.scripts.test_rviz_visualization:main',
        ],
    },
)

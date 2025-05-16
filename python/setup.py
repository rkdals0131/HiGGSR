from setuptools import setup, find_packages

with open("../README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="HiGGSR",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.19.0",
        "open3d>=0.12.0",
        "matplotlib>=3.3.0",
        "scipy>=1.5.0",
        "numba>=0.50.0",
    ],
    author="Gangmin Kim",
    author_email="kikiws70@gmail.com",
    description="Hierarchical Global Grid Search and Registration for 3D Point Clouds",
    long_description=long_description,
    long_description_content_type="text/markdown",
    keywords="point-cloud, registration, 3d-vision, robotics, lidar",
    url="https://github.com/rkdals0131/HiGGSR",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires='>=3.6',
    include_package_data=True,
    entry_points={
        'console_scripts': [
            'higgsr=HiGGSR.main:main',
        ],
    },
) 
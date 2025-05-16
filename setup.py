from setuptools import setup, find_packages

setup(
    name="HiGGSR",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "open3d",
        "matplotlib",
        "scipy",
        "numba",
    ],
    author="Gangmin Kim",
    author_email="kikiws70@gmail.com",
    description="Hierarchical Global Grid Search and Registration for 3D Point Clouds",
    keywords="point-cloud, registration, 3d-vision, robotics",
    url="https://github.com/rkdals0131/HiGGSR",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering",
    ],
    python_requires='>=3.6',
) 
from setuptools import setup, find_packages

setup(
    name="calculate",
    version="1.0.0",
    description="计算文档指标和综合得分",
    author="Mingyu Yuan",
    author_email="yuanmingyu666@gmail.com",
    packages=find_packages(),
    install_requires=[
        "pandas>=1.3.0",
        "numpy>=1.20.0",
        "scikit-learn>=1.0.0",
        "scipy>=1.7.0",
        "tqdm>=4.60.0",
        "chardet>=4.0.0",
    ],
    extras_require={
        "gpu": ["cupy-cuda12x>=10.0.0"],
        "dev": ["pytest>=6.2.5", "pylint>=2.10.0"],
    },
    entry_points={
        "console_scripts": [
            "calculate=src.main:main",
        ],
    },
    python_requires=">=3.7",
)

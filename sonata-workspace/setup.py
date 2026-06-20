"""
Setup script for Sonata-LiDiff
"""

from setuptools import setup, find_packages
import os

# Read README
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [
        line.strip() for line in fh 
        if line.strip() and not line.startswith("#")
    ]

setup(
    name="sonata-lidiff",
    version="0.1.0",
    author="Mikhail Sannikov",
    author_email="msannikov03@icloud.com",
    description="CloudDiffusion: Semantic Scene Completion with a Sonata Encoder and coordinate-space diffusion",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/msannikov03/sonata_ws",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.9",  # spconv-cu124 requires Python >=3.9
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "black>=21.0",
            "flake8>=3.9.0",
            "isort>=5.9.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "train-sonata-lidiff=training.train_diffusion:main",
            "train-refinement=training.train_refinement:main",
        ],
    },
)

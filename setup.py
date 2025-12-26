#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
DeQoG - Diversity-Driven Quality-Assured Code Generation
Setup script for the DeQoG package.
"""

from setuptools import setup, find_packages
import os

# Read the README file
def read_readme():
    with open("README.md", "r", encoding="utf-8") as f:
        return f.read()

# Read requirements
def read_requirements():
    with open("requirements.txt", "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="deqog",
    version="1.0.0",
    author="DeQoG Team",
    author_email="deqog@example.com",
    description="Diversity-Driven Quality-Assured Code Generation for Fault-Tolerant N-Version Programming",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/deqog/deqog",
    project_urls={
        "Bug Tracker": "https://github.com/deqog/deqog/issues",
        "Documentation": "https://deqog.readthedocs.io/",
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Code Generators",
    ],
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.9",
    install_requires=[
        "openai>=1.12.0",
        "anthropic>=0.18.0",
        "transformers>=4.36.0",
        "torch>=2.1.0",
        "sentence-transformers>=2.2.2",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scikit-learn>=1.3.0",
        "pyyaml>=6.0",
        "python-dotenv>=1.0.0",
        "tqdm>=4.66.0",
        "rich>=13.5.0",
        "click>=8.1.0",
        "loguru>=0.7.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-asyncio>=0.21.0",
            "pytest-cov>=4.1.0",
            "black>=23.0.0",
            "pylint>=3.0.0",
        ],
        "notebook": [
            "jupyter>=1.0.0",
            "matplotlib>=3.8.0",
            "seaborn>=0.13.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "deqog=deqog.cli:main",
        ],
    },
)


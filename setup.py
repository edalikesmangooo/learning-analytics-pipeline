"""
Setup configuration for Learning Analytics Pipeline
"""

from setuptools import setup, find_packages

def read_readme():
    try:
        with open("README.md", "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return "A comprehensive framework for behavioral change detection in educational games"

def read_requirements():
    try:
        with open("requirements.txt", "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip() and not line.startswith("#")]
    except FileNotFoundError:
        return []

setup(
    name="learning-analytics-pipeline",
    version="1.0.0",
    author="Your Name",  # Replace with your actual name
    author_email="your.email@institution.edu",  # Replace with your email
    description="A comprehensive framework for behavioral change detection in educational games",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/learning-analytics-pipeline",  # Replace with your GitHub
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Education",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Education",
        "Topic :: Scientific/Engineering :: Information Analysis",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0", 
            "black>=22.0.0",
            "flake8>=4.0.0",
        ],
    },
    keywords=[
        "learning analytics",
        "educational games", 
        "behavioral analysis",
        "change point detection",
        "educational technology",
    ],
)
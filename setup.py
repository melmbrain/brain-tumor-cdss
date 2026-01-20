"""
Brain Tumor CDSS - Clinical Decision Support System for Brain Tumor Diagnosis

Installation:
    pip install brain-tumor-cdss

Development installation:
    pip install -e .[dev]
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding="utf-8")

# Read requirements
requirements = []
requirements_path = this_directory / "requirements.txt"
if requirements_path.exists():
    requirements = requirements_path.read_text().strip().split("\n")
    requirements = [r.strip() for r in requirements if r.strip() and not r.startswith("#")]

setup(
    name="brain-tumor-cdss",
    version="1.0.0",
    author="melmbrain",
    author_email="melmbrain@example.com",
    description="Clinical Decision Support System for Brain Tumor Diagnosis using Multimodal Deep Learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/melmbrain/brain-tumor-cdss",
    project_urls={
        "Bug Tracker": "https://github.com/melmbrain/brain-tumor-cdss/issues",
        "Documentation": "https://melmbrain.github.io/brain-tumor-cdss/",
        "Source Code": "https://github.com/melmbrain/brain-tumor-cdss",
    },
    packages=find_packages(exclude=["tests", "tests.*", "notebooks", "docs"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Healthcare Industry",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "Topic :: Scientific/Engineering :: Image Processing",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "isort>=5.12.0",
            "mypy>=1.0.0",
        ],
        "docs": [
            "mkdocs>=1.5.0",
            "mkdocs-material>=9.0.0",
            "pymdown-extensions>=10.0.0",
        ],
        "api": [
            "fastapi>=0.100.0",
            "uvicorn>=0.23.0",
            "python-multipart>=0.0.6",
        ],
    },
    entry_points={
        "console_scripts": [
            "brain-tumor-cdss=inference.predict:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.yaml", "*.yml", "*.json"],
    },
    keywords=[
        "brain tumor",
        "glioma",
        "deep learning",
        "medical imaging",
        "MRI",
        "gene expression",
        "clinical decision support",
        "multimodal",
        "transformer",
        "SwinUNETR",
    ],
)

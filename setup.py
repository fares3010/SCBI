"""
Setup script for SCBI (Stochastic Covariance-Based Initialization)
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="scbi-init",
    version="1.0.0",
    author="Fares Ashraf",
    author_email="farsashraf44@gmail.com",  
    description="A novel neural network weight initialization method achieving 87× faster convergence",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/fares3010/SCBI",  
    py_modules=["scbi"],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.7",
    install_requires=[
        "torch>=1.9.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "matplotlib>=3.3.0",
            "scikit-learn>=0.24.0",
        ],
    },
    keywords="neural-network initialization deep-learning pytorch machine-learning",
    project_urls={
        "Bug Reports": "https://github.com/fares3010/SCBI/blob/main/issues",
        "Source": "https://github.com/fares3010/SCBI",
        "Documentation": "https://github.com/fares3010/SCBI/blob/main/README.md",
    },
)

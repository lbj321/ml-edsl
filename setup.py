from setuptools import setup, find_packages

setup(
    name="mlir-edsl",
    version="0.4.0",
    description="MLIR-based Embedded Domain-Specific Language for Machine Learning with C++ backend, LLVM lowering, and JIT execution",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "pybind11>=2.10.0",
        "numpy>=1.20.0",
        "protobuf>=4.21.0",  # Must match CMake-built protobuf version
    ],
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "black>=22.0.0",
        ]
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Compilers",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: C++",
    ],
    long_description="""
    ML-EDSL is a Python frontend with C++ MLIR backend for machine learning computations.
    
    Features:
    - Python API for defining ML computations
    - C++ MLIR backend for efficient IR generation
    - LLVM lowering with optimization passes
    - JIT execution for high performance
    - Type-aware operations (int32, float32) with automatic promotion
    
    Architecture: Python frontend → C++ MLIR → LLVM IR → JIT execution
    """,
    long_description_content_type="text/plain",
)
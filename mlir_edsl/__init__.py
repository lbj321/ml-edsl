"""
MLIR-based Embedded Domain-Specific Language for Machine Learning

Phase 3: C++ MLIR Backend
- C++ MLIR IR generation with pybind11 bindings
- Real MLIR objects instead of string-based generation
- Type-aware operations with automatic promotion
- Fallback to string-based generation if C++ backend unavailable
"""

from .types import I32, F32, I1, i32, f32, i1
from .ops import add, sub, mul, div, lt, le, gt, ge, eq, ne, If, For, While, cast
from .functions import ml_function
from .ast import Value, Constant, BinaryOp, CastOp

# Conditionally import C++ backend
try:
    from .backend import get_backend, HAS_CPP_BACKEND
    __all_cpp__ = ["get_backend", "HAS_CPP_BACKEND"]
except ImportError:
    __all_cpp__ = []

__version__ = "0.3.0"

__all__ = [
    # Main decorator
    "ml_function",

    # Type system
    "I32", "F32", "I1", "i32", "f32", "i1",

    # Arithmetic operations
    "add", "sub", "mul", "div", "cast",

    # Comparison operations
    "lt", "le", "gt", "ge", "eq", "ne",

    # Control flow
    "If", "For", "While",

    # AST classes (for advanced usage)
    "Value",
    "Constant",
    "BinaryOp",
    "CastOp",
] + __all_cpp__
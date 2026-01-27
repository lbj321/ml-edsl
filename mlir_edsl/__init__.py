"""
MLIR-based Embedded Domain-Specific Language for Machine Learning

Phase 3: C++ MLIR Backend
- C++ MLIR IR generation with pybind11 bindings
- Real MLIR objects instead of string-based generation
- Type-aware operations with automatic promotion
- Fallback to string-based generation if C++ backend unavailable
"""

from .types import i32, f32, i1, Array
from .ops import add, sub, mul, div, lt, le, gt, ge, eq, ne, If, For, While, cast, call
from .functions import ml_function
from .ast import Value, Constant, BinaryOp, CastOp, ArrayLiteral, ArrayAccess, ArrayStore
from . import ast_pb2

# Binary operation constants (from protobuf schema)
ADD = ast_pb2.ADD
SUB = ast_pb2.SUB
MUL = ast_pb2.MUL
DIV = ast_pb2.DIV

# Comparison predicate constants (from protobuf schema)
# Signed integer comparisons
SLT = ast_pb2.SLT
SLE = ast_pb2.SLE
SGT = ast_pb2.SGT
SGE = ast_pb2.SGE
EQ = ast_pb2.EQ
NE = ast_pb2.NE

# Ordered float comparisons
OLT = ast_pb2.OLT
OLE = ast_pb2.OLE
OGT = ast_pb2.OGT
OGE = ast_pb2.OGE
OEQ = ast_pb2.OEQ
ONE = ast_pb2.ONE

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
    "i32", "f32", "i1",  # Scalar types
    "Array",              # Array type constructor

    # Arithmetic operations
    "add", "sub", "mul", "div", "cast",

    # Comparison operations
    "lt", "le", "gt", "ge", "eq", "ne",

    # Control flow
    "If", "For", "While",

    # Function calls (for recursion)
    "call",

    # Operation constants (for For/While loops)
    "ADD", "SUB", "MUL", "DIV",

    # Predicate constants (for While loops)
    "SLT", "SLE", "SGT", "SGE",  # Signed integer
    "EQ", "NE",                   # Equality
    "OLT", "OLE", "OGT", "OGE", "OEQ", "ONE",  # Ordered float

    # AST classes (for advanced usage)
    "Value",
    "Constant",
    "BinaryOp",
    "CastOp",
    "ArrayLiteral",
    "ArrayAccess",
    "ArrayStore",
] + __all_cpp__

"""User-facing functional API for MLIR operations"""

from .arithmetic import add, sub, mul, div
from .comparison import lt, le, gt, ge, eq, ne
from .control_flow import If, For
from .conversion import cast, call
from .linalg import dot, matmul

__all__ = [
    # Arithmetic
    'add', 'sub', 'mul', 'div',
    # Comparison
    'lt', 'le', 'gt', 'ge', 'eq', 'ne',
    # Control flow
    'If', 'For',
    # Utilities
    'cast', 'call',
    # Linear algebra
    'dot', 'matmul',
]

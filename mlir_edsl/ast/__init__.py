"""Abstract Syntax Tree for MLIR EDSL"""

# Core infrastructure
from .base import Value
from .serialization import SerializationContext
from .helpers import to_value

# AST nodes (re-exported from nodes/)
from .nodes import (
    # Scalars
    Constant, BinaryOp, CompareOp, CastOp,
    # Arrays
    ArrayLiteral, ArrayAccess, ArrayStore, ArrayBinaryOp,
    # Control flow
    IfOp, ForLoopOp, WhileLoopOp,
    # Functions
    Parameter, CallOp,
)

__all__ = [
    'Value', 'SerializationContext', 'to_value',
    'Constant', 'BinaryOp', 'CompareOp', 'CastOp',
    'ArrayLiteral', 'ArrayAccess', 'ArrayStore', 'ArrayBinaryOp',
    'IfOp', 'ForLoopOp', 'WhileLoopOp',
    'Parameter', 'CallOp',
]

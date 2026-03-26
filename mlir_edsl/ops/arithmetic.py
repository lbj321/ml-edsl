"""Arithmetic operations with scalar/array dispatch

All operations (add, sub, mul, div) automatically dispatch to:
- BinaryOp for scalar operands
- ArrayBinaryOp for array operands (element-wise)
"""

from ..ast import BinaryOp, Value, ArrayBinaryOp, to_value
from ..ast.nodes.linalg import LinalgBinaryOp
from ..types import ArrayType, ScalarType, TensorType
from .. import ast_pb2


def _dispatch_binary_op(op: int, left: Value, right: Value):
    left_type = left.infer_type()
    right_type = right.infer_type()
    if isinstance(left_type, TensorType) or isinstance(right_type, TensorType):
        return LinalgBinaryOp(op, left, right)
    elif isinstance(left_type, ArrayType) or isinstance(right_type, ArrayType):
        return ArrayBinaryOp(op, left, right)
    elif isinstance(left_type, ScalarType) and isinstance(right_type, ScalarType):
        return BinaryOp(op, left, right)
    else:
        raise TypeError(
            f"Arithmetic not supported for operand types "
            f"{left_type} and {right_type}."
        )


def add(left, right):
    return _dispatch_binary_op(ast_pb2.ADD, to_value(left), to_value(right))


def sub(left, right):
    return _dispatch_binary_op(ast_pb2.SUB, to_value(left), to_value(right))


def mul(left, right):
    return _dispatch_binary_op(ast_pb2.MUL, to_value(left), to_value(right))


def div(left, right):
    return _dispatch_binary_op(ast_pb2.DIV, to_value(left), to_value(right))

"""Arithmetic operations with scalar/array dispatch

All operations (add, sub, mul, div) automatically dispatch to:
- BinaryOp for scalar operands
- ArrayBinaryOp for array operands (element-wise)
"""

from ..ast import BinaryOp, Value, ArrayBinaryOp, to_value
from ..types import ArrayType
from .. import ast_pb2


def _dispatch_binary_op(op: int, left: Value, right: Value):
    left_is_array = isinstance(left.infer_type(), ArrayType)
    right_is_array = isinstance(right.infer_type(), ArrayType)

    if left_is_array or right_is_array:
        return ArrayBinaryOp(op, left, right)
    else:
        return BinaryOp(op, left, right)


def add(left, right):
    return _dispatch_binary_op(ast_pb2.ADD, to_value(left), to_value(right))


def sub(left, right):
    return _dispatch_binary_op(ast_pb2.SUB, to_value(left), to_value(right))


def mul(left, right):
    return _dispatch_binary_op(ast_pb2.MUL, to_value(left), to_value(right))


def div(left, right):
    return _dispatch_binary_op(ast_pb2.DIV, to_value(left), to_value(right))

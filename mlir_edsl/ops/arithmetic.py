"""Arithmetic operations with scalar/array dispatch

All operations (add, sub, mul, div) automatically dispatch to:
- BinaryOp for scalar operands
- ArrayBinaryOp for array operands (element-wise)
"""

from typing import Union
from ..ast import BinaryOp, Constant, Value, ArrayBinaryOp
from ..types import ArrayType
from .. import ast_pb2


def _to_value(x) -> Value:
    return x if isinstance(x, Value) else Constant(x)


def _dispatch_binary_op(op: int, left: Value, right: Value):
    left_is_array = isinstance(left.infer_type(), ArrayType)
    right_is_array = isinstance(right.infer_type(), ArrayType)

    if left_is_array or right_is_array:
        return ArrayBinaryOp(op, left, right)
    else:
        return BinaryOp(op, left, right)


def add(left, right):
    return _dispatch_binary_op(ast_pb2.ADD, _to_value(left), _to_value(right))


def sub(left, right):
    return _dispatch_binary_op(ast_pb2.SUB, _to_value(left), _to_value(right))


def mul(left, right):
    return _dispatch_binary_op(ast_pb2.MUL, _to_value(left), _to_value(right))


def div(left, right):
    return _dispatch_binary_op(ast_pb2.DIV, _to_value(left), _to_value(right))

"""Comparison operations with predicate inference

Maps base predicates to their float/int variants:
- Float comparisons use ordered predicates (OLT, OLE, OGT, OGE, OEQ, ONE)
- Integer comparisons use signed predicates (SLT, SLE, SGT, SGE, EQ, NE)
"""

from ..ast import CompareOp, Constant, Value
from .. import ast_pb2


def _infer_predicate(float_pred: int, int_pred: int, left_type, right_type) -> int:
    is_float = left_type.is_float() or right_type.is_float()
    return float_pred if is_float else int_pred


def _to_value(x):
    return x if isinstance(x, Value) else Constant(x)


def _make_compare(float_pred: int, int_pred: int, left, right) -> CompareOp:
    left = _to_value(left)
    right = _to_value(right)
    predicate = _infer_predicate(float_pred, int_pred, left.infer_type(), right.infer_type())
    return CompareOp(predicate, left, right)


def lt(left, right):
    """Less than: left < right"""
    return _make_compare(ast_pb2.OLT, ast_pb2.SLT, left, right)


def le(left, right):
    """Less than or equal: left <= right"""
    return _make_compare(ast_pb2.OLE, ast_pb2.SLE, left, right)


def gt(left, right):
    """Greater than: left > right"""
    return _make_compare(ast_pb2.OGT, ast_pb2.SGT, left, right)


def ge(left, right):
    """Greater than or equal: left >= right"""
    return _make_compare(ast_pb2.OGE, ast_pb2.SGE, left, right)


def eq(left, right):
    """Equal: left == right"""
    return _make_compare(ast_pb2.OEQ, ast_pb2.EQ, left, right)


def ne(left, right):
    """Not equal: left != right"""
    return _make_compare(ast_pb2.ONE, ast_pb2.NE, left, right)

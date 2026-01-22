"""Comparison operations with predicate inference"""

from typing import Union
from ..ast import CompareOp, Constant, Value


def _infer_predicate(base: str, left_type, right_type) -> str:
    """Infer comparison predicate based on operand types

    Maps base predicates to their float/int variants:
    - Float comparisons use 'o' prefix (olt, ole, ogt, oge, oeq, one)
    - Integer comparisons use 's' prefix for signed (slt, sle, sgt, sge)
    - Equality uses 'eq' and 'ne' for integers
    """
    predicates = {
        'lt': ('olt', 'slt'),
        'le': ('ole', 'sle'),
        'gt': ('ogt', 'sgt'),
        'ge': ('oge', 'sge'),
        'eq': ('oeq', 'eq'),
        'ne': ('one', 'ne'),
    }

    float_pred, int_pred = predicates[base]
    is_float = left_type.is_float() or right_type.is_float()
    return float_pred if is_float else int_pred


def lt(left: Union[int, float, Value], right: Union[int, float, Value]) -> CompareOp:
    """Create a less-than comparison (left < right)

    Automatically uses signed integer comparison (slt) for ints
    or ordered float comparison (olt) for floats.

    Args:
        left: Left operand (integer, float, or Value)
        right: Right operand (integer, float, or Value)

    Returns:
        CompareOp representing left < right
    """
    if isinstance(left, (int, float)):
        left = Constant(left)
    if isinstance(right, (int, float)):
        right = Constant(right)

    # Infer predicate based on operand types
    left_type = left.infer_type()
    right_type = right.infer_type()
    predicate = _infer_predicate('lt', left_type, right_type)

    return CompareOp(predicate, left, right)


def le(left: Union[int, float, Value], right: Union[int, float, Value]) -> CompareOp:
    """Create a less-than-or-equal comparison (left <= right)

    Args:
        left: Left operand (integer, float, or Value)
        right: Right operand (integer, float, or Value)

    Returns:
        CompareOp representing left <= right
    """
    if isinstance(left, (int, float)):
        left = Constant(left)
    if isinstance(right, (int, float)):
        right = Constant(right)

    left_type = left.infer_type()
    right_type = right.infer_type()
    predicate = _infer_predicate('le', left_type, right_type)

    return CompareOp(predicate, left, right)


def gt(left: Union[int, float, Value], right: Union[int, float, Value]) -> CompareOp:
    """Create a greater-than comparison (left > right)

    Args:
        left: Left operand (integer, float, or Value)
        right: Right operand (integer, float, or Value)

    Returns:
        CompareOp representing left > right
    """
    if isinstance(left, (int, float)):
        left = Constant(left)
    if isinstance(right, (int, float)):
        right = Constant(right)

    left_type = left.infer_type()
    right_type = right.infer_type()
    predicate = _infer_predicate('gt', left_type, right_type)

    return CompareOp(predicate, left, right)


def ge(left: Union[int, float, Value], right: Union[int, float, Value]) -> CompareOp:
    """Create a greater-than-or-equal comparison (left >= right)

    Args:
        left: Left operand (integer, float, or Value)
        right: Right operand (integer, float, or Value)

    Returns:
        CompareOp representing left >= right
    """
    if isinstance(left, (int, float)):
        left = Constant(left)
    if isinstance(right, (int, float)):
        right = Constant(right)

    left_type = left.infer_type()
    right_type = right.infer_type()
    predicate = _infer_predicate('ge', left_type, right_type)

    return CompareOp(predicate, left, right)


def eq(left: Union[int, float, Value], right: Union[int, float, Value]) -> CompareOp:
    """Create an equality comparison (left == right)

    Args:
        left: Left operand (integer, float, or Value)
        right: Right operand (integer, float, or Value)

    Returns:
        CompareOp representing left == right
    """
    if isinstance(left, (int, float)):
        left = Constant(left)
    if isinstance(right, (int, float)):
        right = Constant(right)

    left_type = left.infer_type()
    right_type = right.infer_type()
    predicate = _infer_predicate('eq', left_type, right_type)

    return CompareOp(predicate, left, right)


def ne(left: Union[int, float, Value], right: Union[int, float, Value]) -> CompareOp:
    """Create a not-equal comparison (left != right)

    Args:
        left: Left operand (integer, float, or Value)
        right: Right operand (integer, float, or Value)

    Returns:
        CompareOp representing left != right
    """
    if isinstance(left, (int, float)):
        left = Constant(left)
    if isinstance(right, (int, float)):
        right = Constant(right)

    left_type = left.infer_type()
    right_type = right.infer_type()
    predicate = _infer_predicate('ne', left_type, right_type)

    return CompareOp(predicate, left, right)

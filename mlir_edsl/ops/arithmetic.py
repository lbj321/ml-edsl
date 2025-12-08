"""Arithmetic operations with scalar/array dispatch"""

from typing import Union
from ..ast import BinaryOp, Constant, Value, ArrayBinaryOp
from ..types import ArrayType


def _to_value(x: Union[int, float, Value]) -> Value:
    """Convert Python literal to Constant if needed"""
    return Constant(x) if isinstance(x, (int, float)) else x


def _dispatch_binary_op(op: str, left: Value, right: Value):
    """Dispatch to BinaryOp or ArrayBinaryOp based on operand types"""
    left_is_array = isinstance(left.infer_type(), ArrayType)
    right_is_array = isinstance(right.infer_type(), ArrayType)

    if left_is_array or right_is_array:
        return ArrayBinaryOp(op, left, right)
    else:
        return BinaryOp(op, left, right)


def add(left: Union[int, float, Value], right: Union[int, float, Value]) -> Union[BinaryOp, ArrayBinaryOp]:
    """Create an addition operation

    Automatically dispatches to:
    - BinaryOp: For scalar + scalar
    - ArrayBinaryOp: For array operations (array + array, array + scalar, scalar + array)

    Args:
        left: Left operand (integer, float, or Value)
        right: Right operand (integer, float, or Value)

    Returns:
        BinaryOp or ArrayBinaryOp depending on operand types
    """
    return _dispatch_binary_op("add", _to_value(left), _to_value(right))


def sub(left: Union[int, float, Value], right: Union[int, float, Value]) -> Union[BinaryOp, ArrayBinaryOp]:
    """Create a subtraction operation

    Automatically dispatches to:
    - BinaryOp: For scalar - scalar
    - ArrayBinaryOp: For array operations (array - array, array - scalar, scalar - array)

    Args:
        left: Left operand (integer, float, or Value)
        right: Right operand (integer, float, or Value)

    Returns:
        BinaryOp or ArrayBinaryOp depending on operand types
    """
    return _dispatch_binary_op("sub", _to_value(left), _to_value(right))


def mul(left: Union[int, float, Value], right: Union[int, float, Value]) -> Union[BinaryOp, ArrayBinaryOp]:
    """Create a multiplication operation

    Automatically dispatches to:
    - BinaryOp: For scalar * scalar
    - ArrayBinaryOp: For array operations (array * array, array * scalar, scalar * array)

    Args:
        left: Left operand (integer, float, or Value)
        right: Right operand (integer, float, or Value)

    Returns:
        BinaryOp or ArrayBinaryOp depending on operand types
    """
    return _dispatch_binary_op("mul", _to_value(left), _to_value(right))


def div(left: Union[int, float, Value], right: Union[int, float, Value]) -> Union[BinaryOp, ArrayBinaryOp]:
    """Create a division operation

    Automatically dispatches to:
    - BinaryOp: For scalar / scalar
    - ArrayBinaryOp: For array operations (array / array, array / scalar, scalar / array)

    Args:
        left: Left operand (integer, float, or Value)
        right: Right operand (integer, float, or Value)

    Returns:
        BinaryOp or ArrayBinaryOp depending on operand types
    """
    return _dispatch_binary_op("div", _to_value(left), _to_value(right))

"""Basic arithmetic operations for the EDSL"""

from typing import Union
from .ast import BinaryOp, Constant, Value


def add(left: Union[int, float, Value], right: Union[int, float, Value]) -> BinaryOp:
    """Create an addition operation
    
    Args:
        left: Left operand (integer, float, or Value)
        right: Right operand (integer, float, or Value)
        
    Returns:
        BinaryOp representing the addition
    """

    if isinstance(left, (int, float)):
        left = Constant(left)
    if isinstance(right, (int, float)):
        right = Constant(right)

    return BinaryOp("add", left, right)


def sub(left: Union[int, float, Value], right: Union[int, float, Value]) -> BinaryOp:
    """Create a subtraction operation
    
    Args:
        left: Left operand (integer, float, or Value)
        right: Right operand (integer, float, or Value)
        
    Returns:
        BinaryOp representing the subtraction
    """

    if isinstance(left, (int, float)):
        left = Constant(left)
    if isinstance(right, (int, float)):
        right = Constant(right)

    return BinaryOp("sub", left, right)


def mul(left: Union[int, float, Value], right: Union[int, float, Value]) -> BinaryOp:
    """Create a multiplication operation
    
    Args:
        left: Left operand (integer, float, or Value)
        right: Right operand (integer, float, or Value)
        
    Returns:
        BinaryOp representing the multiplication
    """

    if isinstance(left, (int, float)):
        left = Constant(left)
    if isinstance(right, (int, float)):
        right = Constant(right)

    return BinaryOp("mul", left, right)


def div(left: Union[int, float, Value], right: Union[int, float, Value]) -> BinaryOp:
    """Create a division operation
    
    Args:
        left: Left operand (integer, float, or Value)
        right: Right operand (integer, float, or Value)
        
    Returns:
        BinaryOp representing the division
    """

    if isinstance(left, (int, float)):
        left = Constant(left)
    if isinstance(right, (int, float)):
        right = Constant(right)

    return BinaryOp("div", left, right)
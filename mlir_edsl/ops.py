"""Basic arithmetic operations for the EDSL"""

from typing import Union
from .ast import BinaryOp, Constant, Value, CallOp, CompareOp
from .types import I32


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


def call(func_name: str, args: list[Union[int, float, Value]], return_type) -> CallOp:
    """Create a function call operation

    Args:
        func_name: Name of the function to call
        args: List of arguments (can be integers, floats, or Values)
        return_type: Expected return type (I32, F32, or I1 enum) - required

    Returns:
        CallOp representing the function call
    """
    # Convert primitive types to Constants
    converted_args = []
    for arg in args:
        if isinstance(arg, (int, float)):
            converted_args.append(Constant(arg))
        else:
            converted_args.append(arg)

    return CallOp(func_name, converted_args, return_type)
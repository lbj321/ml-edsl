"""Type conversion and function call utilities"""

from typing import Union
from ..ast import CastOp, CallOp, Value, to_value


def cast(value: Union[int, float, Value], target_type) -> CastOp:
    """Explicitly cast a value to a different type

    Args:
        value: Value to cast (int, float, or AST Value)
        target_type: Target MLIR type (i32, f32, i1)

    Returns:
        CastOp representing the explicit conversion

    Examples:
        cast(my_float, i32)  # f32 -> i32 (truncate)
        cast(my_int, f32)    # i32 -> f32 (convert)
        cast(5.7, i32)       # Constant 5.7 -> i32
        result = add(cast(a, f32), b)  # a is i32, b is f32
    """

    return CastOp(to_value(value), target_type)


def call(func_name: str, args: list[Union[int, float, Value]], return_type) -> CallOp:
    """Create a function call operation

    Args:
        func_name: Name of the function to call
        args: List of arguments (can be integers, floats, or Values)
        return_type: Expected return type (i32, f32, i1) - required

    Returns:
        CallOp representing the function call
    """
    # Convert primitive types to Constants
    converted_args = []
    for arg in args:
        converted_args.append(to_value(arg))

    return CallOp(func_name, converted_args, return_type)

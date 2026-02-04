"""Control flow operation helpers"""

from typing import Union
from ..ast import IfOp, ForLoopOp, Value, to_value
from .. import ast_pb2


def If(condition: Value, then_value: Union[int, float, Value],
       else_value: Union[int, float, Value]):
    """Create a conditional if-else expression

    Args:
        condition: Boolean condition (must be CompareOp or other I1 value)
        then_value: Value to return if condition is true
        else_value: Value to return if condition is false

    Returns:
        IfOp representing the conditional

    Example:
        @ml_function
        def max_value(x, y):
            return If(x > y, then_value=x, else_value=y)
    """

    return IfOp(condition, to_value(then_value), to_value(else_value))


def For(start: Union[int, Value],
        end: Union[int, Value],
        init: Union[int, float, Value],
        operation: int = ast_pb2.ADD,
        step: Union[int, Value] = 1):
    """Create a for loop: for(i = start; i < end; i += step) { accumulator op= i }

    Args:
        start: Loop start value (inclusive)
        end: Loop end value (exclusive)
        init: Initial accumulator value
        operation: Operation enum - ast_pb2.ADD, SUB, MUL, or DIV
        step: Loop step increment (default: 1)

    Returns:
        ForLoopOp representing the loop

    Examples:
        # Sum from 0 to 9: result = 0 + 0 + 1 + 2 + ... + 9
        For(start=0, end=10, init=0, operation=ast_pb2.ADD)

        # Factorial-like: result = 1 * 1 * 2 * 3 * 4
        For(start=1, end=5, init=1, operation=ast_pb2.MUL)

        # With step: sum even numbers 0 to 8
        For(start=0, end=10, init=0, operation=ast_pb2.ADD, step=2)
    """

    return ForLoopOp(to_value(start), to_value(end), to_value(step), to_value(init), operation)

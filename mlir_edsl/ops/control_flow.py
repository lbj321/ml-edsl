"""Control flow operation helpers"""

from typing import Union
from ..ast import IfOp, ForLoopOp, WhileLoopOp, Value, to_value
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


def While(init: Union[int, float, Value],
          target: Union[int, float, Value],
          operation: int = ast_pb2.ADD,
          predicate: int = ast_pb2.SLT):
    """Create a while loop: while(current predicate target) { current = current op step }

    The step value is determined by operation type:
    - ADD/SUB: step = 1
    - MUL/DIV: step = 2

    Args:
        init: Initial loop value
        target: Target value for comparison
        operation: Operation enum - ast_pb2.ADD, SUB, MUL, or DIV
        predicate: Comparison predicate enum:
            - ast_pb2.SLT/SGT/SLE/SGE for signed integers
            - ast_pb2.OLT/OGT/OLE/OGE for ordered floats
            - ast_pb2.EQ/NE for equality checks

    Returns:
        WhileLoopOp representing the loop

    Examples:
        # Count up: while(i < 10) { i = i + 1 }
        While(init=0, target=10, operation=ast_pb2.ADD, predicate=ast_pb2.SLT)

        # Count down: while(i > 0) { i = i - 1 }
        While(init=10, target=0, operation=ast_pb2.SUB, predicate=ast_pb2.SGT)

        # Double until: while(i < 16) { i = i * 2 }
        While(init=1, target=16, operation=ast_pb2.MUL, predicate=ast_pb2.SLT)

        # Until equal: while(i != 5) { i = i + 1 }
        While(init=0, target=5, operation=ast_pb2.ADD, predicate=ast_pb2.NE)
    """

    return WhileLoopOp(to_value(init), to_value(target), operation, predicate)

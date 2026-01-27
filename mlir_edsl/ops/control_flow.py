"""Control flow operation helpers"""

from typing import Union
from ..ast import IfOp, ForLoopOp, WhileLoopOp, Constant, Value
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
    # Convert primitives to Constants
    if isinstance(then_value, (int, float)):
        then_value = Constant(then_value)
    if isinstance(else_value, (int, float)):
        else_value = Constant(else_value)

    return IfOp(condition, then_value, else_value)


def For(start: Union[int, Value],
        end: Union[int, Value],
        init: Union[int, float, Value],
        operation: str = "add",
        step: Union[int, Value] = 1):
    """Create a for loop: for(i = start; i < end; i += step) { accumulator op= i }

    Args:
        start: Loop start value (inclusive)
        end: Loop end value (exclusive)
        init: Initial accumulator value
        operation: Operation to perform - "add", "mul", "sub", or "div"
        step: Loop step increment (default: 1)

    Returns:
        ForLoopOp representing the loop

    Examples:
        # Sum from 0 to 9: result = 0 + 0 + 1 + 2 + ... + 9
        For(start=0, end=10, init=0, operation="add")

        # Factorial-like: result = 1 * 1 * 2 * 3 * 4
        For(start=1, end=5, init=1, operation="mul")

        # With step: sum even numbers 0 to 8
        For(start=0, end=10, init=0, operation="add", step=2)
    """
    # Convert primitives to Constants
    if isinstance(start, (int, float)):
        start = Constant(start)
    if isinstance(end, (int, float)):
        end = Constant(end)
    if isinstance(step, (int, float)):
        step = Constant(step)
    if isinstance(init, (int, float)):
        init = Constant(init)

    return ForLoopOp(start, end, step, init, operation)


def While(init: Union[int, float, Value],
          target: Union[int, float, Value],
          operation: str = "add",
          predicate: str = "slt"):
    """Create a while loop: while(current predicate target) { current = current op step }

    The step value is determined by operation type:
    - "add"/"sub": step = 1
    - "mul"/"div": step = 2

    Args:
        init: Initial loop value
        target: Target value for comparison
        operation: Operation to perform - "add", "mul", "sub", or "div"
        predicate: Comparison predicate:
            - "slt"/"sgt"/"sle"/"sge" for integers
            - "olt"/"ogt"/"ole"/"oge" for floats
            - "eq"/"ne" for equality checks

    Returns:
        WhileLoopOp representing the loop

    Examples:
        # Count up: while(i < 10) { i = i + 1 }
        While(init=0, target=10, operation="add", predicate="slt")

        # Count down: while(i > 0) { i = i - 1 }
        While(init=10, target=0, operation="sub", predicate="sgt")

        # Double until: while(i < 16) { i = i * 2 }
        While(init=1, target=16, operation="mul", predicate="slt")

        # Until equal: while(i != 5) { i = i + 1 }
        While(init=0, target=5, operation="add", predicate="ne")
    """
    # Convert primitives to Constants
    if isinstance(init, (int, float)):
        init = Constant(init)
    if isinstance(target, (int, float)):
        target = Constant(target)

    return WhileLoopOp(init, target, operation, predicate)

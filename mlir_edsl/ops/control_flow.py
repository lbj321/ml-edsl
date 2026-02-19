"""Control flow operation helpers"""

from typing import Callable, Union
from ..ast import IfOp, ForLoopOp, Value, to_value
from ..ast.nodes.control_flow import ForIndex, ForIterArg


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
        body: Callable[[Value, Value], Value],
        step: Union[int, Value] = 1):
    """Create a for loop with lambda body (scf.for with iter_args).

    Args:
        start: Loop start value (inclusive)
        end: Loop end value (exclusive)
        init: Initial accumulator value (scalar or tensor)
        body: Lambda(index, accumulator) -> new_accumulator
        step: Loop step increment (default: 1)

    Returns:
        ForLoopOp representing the loop result

    Examples:
        # Sum 0..9
        result = For(0, 10, init=0, body=lambda i, acc: acc + i)

        # Fill tensor with i*2
        t = Tensor.empty(i32, 4)
        t = For(0, 4, init=t, body=lambda i, acc: acc.at[i].set(i * 2))
    """
    init_val = to_value(init)
    init_type = init_val.infer_type()

    # Create placeholder nodes
    index_placeholder = ForIndex()
    iter_arg_placeholder = ForIterArg(init_type)

    # Call body lambda to build the AST subtree
    body_result = body(index_placeholder, iter_arg_placeholder)

    # Wrap raw Python values
    if not isinstance(body_result, Value):
        body_result = to_value(body_result)

    return ForLoopOp(
        to_value(start), to_value(end), to_value(step),
        init_val, body_result,
        index_placeholder, iter_arg_placeholder
    )

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


# ==================== COMPARISON OPERATORS ====================

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

    from .types import F32
    # Use float comparison if either operand is float
    predicate = "olt" if (left_type == F32 or right_type == F32) else "slt"

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

    from .types import F32
    left_type = left.infer_type()
    right_type = right.infer_type()
    predicate = "ole" if (left_type == F32 or right_type == F32) else "sle"

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

    from .types import F32
    left_type = left.infer_type()
    right_type = right.infer_type()
    predicate = "ogt" if (left_type == F32 or right_type == F32) else "sgt"

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

    from .types import F32
    left_type = left.infer_type()
    right_type = right.infer_type()
    predicate = "oge" if (left_type == F32 or right_type == F32) else "sge"

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

    from .types import F32
    left_type = left.infer_type()
    right_type = right.infer_type()
    predicate = "oeq" if (left_type == F32 or right_type == F32) else "eq"

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

    from .types import F32
    left_type = left.infer_type()
    right_type = right.infer_type()
    predicate = "one" if (left_type == F32 or right_type == F32) else "ne"

    return CompareOp(predicate, left, right)


# ==================== CONTROL FLOW HELPERS ====================

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
    from .ast import IfOp

    # Convert primitives to Constants
    if isinstance(then_value, (int, float)):
        then_value = Constant(then_value)
    if isinstance(else_value, (int, float)):
        else_value = Constant(else_value)

    return IfOp(condition, then_value, else_value)


def For(start: Union[int, float, Value],
        end: Union[int, float, Value],
        init: Union[int, float, Value],
        operation: str = "add",
        step: Union[int, float, Value] = 1):
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
    from .ast import ForLoopOp

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
    from .ast import WhileLoopOp

    # Convert primitives to Constants
    if isinstance(init, (int, float)):
        init = Constant(init)
    if isinstance(target, (int, float)):
        target = Constant(target)

    return WhileLoopOp(init, target, operation, predicate)


def cast(value: Union[int, float, Value], target_type) -> 'CastOp':
    """Explicitly cast a value to a different type

    Args:
        value: Value to cast (int, float, or AST Value)
        target_type: Target MLIR type (i32, f32, i1 object or I32, F32, I1 enum)

    Returns:
        CastOp representing the explicit conversion

    Examples:
        cast(my_float, i32)  # f32 -> i32 (truncate)
        cast(my_int, f32)    # i32 -> f32 (convert)
        cast(5.7, i32)       # Constant 5.7 -> i32
        result = add(cast(a, f32), b)  # a is i32, b is f32
    """
    from .ast import CastOp
    from .types import MLIRType

    if isinstance(value, (int, float)):
        value = Constant(value)

    if isinstance(target_type, MLIRType):
        target_type = target_type.enum_value

    return CastOp(value, target_type)
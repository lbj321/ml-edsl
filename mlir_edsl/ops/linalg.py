"""Linalg operations: dot product, matrix multiply, element-wise map, and reductions"""

from ..ast.nodes.linalg import LinalgDot, LinalgMatmul, LinalgMap, LinalgReduce
from ..ast.base import Value


def dot(a: Value, b: Value) -> LinalgDot:
    """Compute dot product of two 1D arrays.

    Args:
        a: 1D array (Array[f32, N] or Array[i32, N])
        b: 1D array of the same element type and length

    Returns:
        LinalgDot AST node; evaluates to a scalar of the element type

    Raises:
        TypeError: if operands are not 1D arrays with matching element types

    Example:
        @ml_function
        def dot_product(a: Array[f32, 4], b: Array[f32, 4]) -> f32:
            return dot(a, b)
    """
    return LinalgDot(a, b)


def matmul(A: Value, B: Value) -> LinalgMatmul:
    """Compute matrix multiplication of two 2D arrays.

    Args:
        A: 2D array of shape [M, K]
        B: 2D array of shape [K, N]

    Returns:
        LinalgMatmul AST node; evaluates to a 2D array of shape [M, N]

    Raises:
        TypeError: if operands are not 2D arrays with compatible shapes/types

    Example:
        @ml_function
        def matrix_mul(A: Array[f32, 4, 4], B: Array[f32, 4, 4]) -> Array[f32, 4, 4]:
            return matmul(A, B)
    """
    return LinalgMatmul(A, B)


def tensor_map(arr: Value, fn) -> LinalgMap:
    """Apply fn element-wise over a 1D array (linalg.map under the hood).

    Args:
        arr: 1D array (Array[f32, N] or Array[i32, N])
        fn: callable accepting a scalar Value placeholder, returning a scalar Value

    Returns:
        LinalgMap AST node; evaluates to a 1D array of the same type

    Raises:
        TypeError: if arr is not a 1D array, or fn returns wrong element type

    Example:
        @ml_function
        def scale(a: Array[f32, 4]) -> Array[f32, 4]:
            return tensor_map(a, lambda v: v * 2.0)
    """
    return LinalgMap(arr, fn)


def relu(arr: Value) -> LinalgMap:
    """ReLU: max(0, x) element-wise over a 1D float array.

    Args:
        arr: 1D array of f32 values

    Returns:
        LinalgMap AST node; evaluates to a 1D array with negative values clamped to 0

    Example:
        @ml_function
        def apply_relu(a: Array[f32, 4]) -> Array[f32, 4]:
            return relu(a)
    """
    from ..ast.nodes.control_flow import IfOp
    from ..ast.helpers import to_value
    return LinalgMap(arr, lambda v: IfOp(v > to_value(0.0), v, to_value(0.0)))


def reduce(arr: Value, init: Value, fn) -> LinalgReduce:
    """Reduce a 1D array to a scalar using a binary combining function.

    Args:
        arr: 1D array (Array[f32, N] or Array[i32, N])
        init: scalar initial accumulator value (same element type as arr)
        fn: callable(element, accumulator) → scalar Value

    Returns:
        LinalgReduce AST node; evaluates to a scalar of the element type

    Raises:
        TypeError: if arr is not 1D, init type mismatches, or fn returns wrong type

    Example:
        @ml_function
        def my_sum(a: Array[f32, 4]) -> f32:
            return reduce(a, to_value(0.0), lambda elem, acc: acc + elem)
    """
    return LinalgReduce(arr, init, fn)


def tensor_sum(arr: Value) -> LinalgReduce:
    """Sum all elements of a 1D float array.

    Args:
        arr: 1D array of f32 values

    Returns:
        LinalgReduce AST node; evaluates to the sum as f32

    Example:
        @ml_function
        def my_sum(a: Array[f32, 4]) -> f32:
            return tensor_sum(a)
    """
    from ..ast.helpers import to_value
    return LinalgReduce(arr, to_value(0.0), lambda elem, acc: acc + elem)


def tensor_max(arr: Value) -> LinalgReduce:
    """Return the maximum element of a 1D float array.

    Uses -infinity as the initial accumulator so any element beats it.

    Args:
        arr: 1D array of f32 values

    Returns:
        LinalgReduce AST node; evaluates to the maximum element as f32

    Example:
        @ml_function
        def my_max(a: Array[f32, 4]) -> f32:
            return tensor_max(a)
    """
    from ..ast.helpers import to_value
    from ..ast.nodes.control_flow import IfOp
    return LinalgReduce(
        arr,
        to_value(float('-inf')),
        lambda elem, acc: IfOp(elem > acc, elem, acc),
    )


def tensor_min(arr: Value) -> LinalgReduce:
    """Return the minimum element of a 1D float array.

    Uses +infinity as the initial accumulator so any element beats it.

    Args:
        arr: 1D array of f32 values

    Returns:
        LinalgReduce AST node; evaluates to the minimum element as f32

    Example:
        @ml_function
        def my_min(a: Array[f32, 4]) -> f32:
            return tensor_min(a)
    """
    from ..ast.helpers import to_value
    from ..ast.nodes.control_flow import IfOp
    return LinalgReduce(
        arr,
        to_value(float('inf')),
        lambda elem, acc: IfOp(elem < acc, elem, acc),
    )


def leaky_relu(arr: Value, alpha: float = 0.01) -> LinalgMap:
    """Leaky ReLU: v if v > 0 else alpha * v, element-wise over a 1D float array.

    Args:
        arr: 1D array of f32 values
        alpha: negative slope coefficient (default 0.01)

    Returns:
        LinalgMap AST node; evaluates to a 1D array with leaky relu applied

    Example:
        @ml_function
        def apply_leaky_relu(a: Array[f32, 4]) -> Array[f32, 4]:
            return leaky_relu(a, alpha=0.1)
    """
    from ..ast.nodes.control_flow import IfOp
    from ..ast.helpers import to_value
    a = float(alpha)
    return LinalgMap(arr, lambda v: IfOp(v > to_value(0.0), v, v * to_value(a)))

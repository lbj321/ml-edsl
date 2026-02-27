"""Linalg operations: dot product and matrix multiply"""

from ..ast.nodes.linalg import LinalgDot, LinalgMatmul
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

"""Tests for aggregate (Array) return types from @ml_function (Phase 8.1)."""
import pytest
from mlir_edsl import ml_function
from mlir_edsl.types import f32, i32, Array


# ==================== BASIC RETURN ====================

def test_identity_1d(backend):
    """Function returns its input array unchanged."""
    @ml_function
    def identity(x: Array[f32, 4]) -> Array[f32, 4]:
        return x

    result = identity([1.0, 2.0, 3.0, 4.0])
    assert result == pytest.approx([1.0, 2.0, 3.0, 4.0])


def test_scalar_multiply_1d(backend):
    """Element-wise scale via array binary op."""
    @ml_function
    def scale(x: Array[f32, 4], factor: f32) -> Array[f32, 4]:
        return x * factor

    result = scale([1.0, 2.0, 3.0, 4.0], 2.0)
    assert result == pytest.approx([2.0, 4.0, 6.0, 8.0])


def test_add_arrays_1d(backend):
    """Element-wise addition of two arrays."""
    @ml_function
    def add_arrays(a: Array[f32, 4], b: Array[f32, 4]) -> Array[f32, 4]:
        return a + b

    result = add_arrays([1.0, 2.0, 3.0, 4.0], [10.0, 20.0, 30.0, 40.0])
    assert result == pytest.approx([11.0, 22.0, 33.0, 44.0])


# ==================== INTEGER ARRAYS ====================

def test_identity_i32(backend):
    """Integer array return."""
    @ml_function
    def identity_i(x: Array[i32, 3]) -> Array[i32, 3]:
        return x

    result = identity_i([10, 20, 30])
    assert result == [10, 20, 30]


# ==================== MULTI-DIMENSIONAL ====================

def test_identity_2d(backend):
    """2D array return — validates _unflatten on multi-dim shape."""
    @ml_function
    def identity_2d(x: Array[f32, 2, 3]) -> Array[f32, 2, 3]:
        return x

    data = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    result = identity_2d(data)
    flat_result = [v for row in result for v in row]
    flat_expected = [v for row in data for v in row]
    assert flat_result == pytest.approx(flat_expected)


def test_scale_2d(backend):
    """2D array element-wise scale."""
    @ml_function
    def scale_2d(x: Array[f32, 2, 2], factor: f32) -> Array[f32, 2, 2]:
        return x * factor

    result = scale_2d([[1.0, 2.0], [3.0, 4.0]], 3.0)
    flat_result = [v for row in result for v in row]
    assert flat_result == pytest.approx([3.0, 6.0, 9.0, 12.0])


# ==================== ERROR CASES ====================

def test_wrong_input_shape(backend):
    """Wrong input shape raises at call time."""
    @ml_function
    def identity(x: Array[f32, 4]) -> Array[f32, 4]:
        return x

    with pytest.raises((TypeError, ValueError)):
        identity([1.0, 2.0, 3.0])  # 3 elements, expects 4

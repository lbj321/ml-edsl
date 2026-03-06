"""Tests for aggregate (Array) return types from @ml_function (Phase 8.1)."""
import pytest
import numpy as np
from mlir_edsl import ml_function
from mlir_edsl.types import f32, i32, Array


# ==================== BASIC RETURN ====================

def test_identity_1d(backend):
    """Function returns its input array unchanged."""
    @ml_function
    def identity(x: Array[f32, 4]) -> Array[f32, 4]:
        return x

    result = identity(np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32))
    assert result == pytest.approx([1.0, 2.0, 3.0, 4.0])


def test_scalar_multiply_1d(backend):
    """Element-wise scale via array binary op."""
    @ml_function
    def scale(x: Array[f32, 4], factor: f32) -> Array[f32, 4]:
        return x * factor

    result = scale(np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32), 2.0)
    assert result == pytest.approx([2.0, 4.0, 6.0, 8.0])


def test_add_arrays_1d(backend):
    """Element-wise addition of two arrays."""
    @ml_function
    def add_arrays(a: Array[f32, 4], b: Array[f32, 4]) -> Array[f32, 4]:
        return a + b

    result = add_arrays(np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
                        np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float32))
    assert result == pytest.approx([11.0, 22.0, 33.0, 44.0])


# ==================== INTEGER ARRAYS ====================

def test_identity_i32(backend):
    """Integer array return."""
    @ml_function
    def identity_i(x: Array[i32, 3]) -> Array[i32, 3]:
        return x

    result = identity_i(np.array([10, 20, 30], dtype=np.int32))
    np.testing.assert_array_equal(result, [10, 20, 30])


# ==================== MULTI-DIMENSIONAL ====================

def test_identity_2d(backend):
    """2D array return — validates shape is preserved in output ndarray."""
    @ml_function
    def identity_2d(x: Array[f32, 2, 3]) -> Array[f32, 2, 3]:
        return x

    data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    result = identity_2d(data)
    np.testing.assert_allclose(result, data, rtol=1e-5)


def test_scale_2d(backend):
    """2D array element-wise scale."""
    @ml_function
    def scale_2d(x: Array[f32, 2, 2], factor: f32) -> Array[f32, 2, 2]:
        return x * factor

    result = scale_2d(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32), 3.0)
    np.testing.assert_allclose(result.flatten(), [3.0, 6.0, 9.0, 12.0], rtol=1e-5)


# ==================== ERROR CASES ====================

def test_wrong_input_shape(backend):
    """Wrong input shape raises at call time."""
    @ml_function
    def identity(x: Array[f32, 4]) -> Array[f32, 4]:
        return x

    with pytest.raises((TypeError, ValueError)):
        identity(np.array([1.0, 2.0, 3.0], dtype=np.float32))  # 3 elements, expects 4

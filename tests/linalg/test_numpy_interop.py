"""Test numpy interoperability (Phase 8.5a)

Validates that np.ndarray can be passed to @ml_function parameters and that
array outputs are returned as np.ndarray.
"""

import pytest
import numpy as np
from mlir_edsl import ml_function, Tensor, f32, i32, tensor_map, tensor_sum
from mlir_edsl.types import DYN


# ==================== INPUT INTEROP ====================

class TestNumpyInput:
    """np.ndarray accepted as input, result is ndarray"""

    def test_scalar_return_from_ndarray_input(self, backend):
        """tensor_sum of a numpy array returns correct scalar."""
        @ml_function
        def my_sum(a: Tensor[f32, 4]) -> f32:
            return tensor_sum(a)

        a = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        result = my_sum(a)
        assert abs(result - 10.0) < 1e-5

    def test_array_return_is_ndarray(self, backend):
        """Array return type comes back as np.ndarray."""
        @ml_function
        def scale(a: Tensor[f32, 4]) -> Tensor[f32, 4]:
            return tensor_map(a, lambda v: v * 2.0)

        a = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        result = scale(a)
        assert isinstance(result, np.ndarray)
        np.testing.assert_allclose(result, [2.0, 4.0, 6.0, 8.0], rtol=1e-5)


# ==================== VALIDATION ====================

class TestNumpyValidation:
    """Wrong dtype or shape raises before calling JIT"""

    def test_wrong_dtype_raises(self):
        """Passing float64 where float32 is expected raises TypeError."""
        @ml_function
        def my_sum(a: Tensor[f32, 4]) -> f32:
            return tensor_sum(a)

        a = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
        with pytest.raises(TypeError, match="dtype"):
            my_sum(a)

    def test_wrong_shape_raises(self):
        """Passing wrong shape raises ValueError."""
        @ml_function
        def my_sum(a: Tensor[f32, 4]) -> f32:
            return tensor_sum(a)

        a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        with pytest.raises(ValueError, match="shape"):
            my_sum(a)

    def test_non_contiguous_accepted(self, backend):
        """Non-contiguous ndarray is auto-copied to contiguous before call."""
        @ml_function
        def my_sum(a: Tensor[f32, 4]) -> f32:
            return tensor_sum(a)

        # Slice every other element from an 8-element array → non-contiguous
        base = np.array([1.0, 0.0, 2.0, 0.0, 3.0, 0.0, 4.0, 0.0], dtype=np.float32)
        a = base[::2]  # [1, 2, 3, 4], non-contiguous
        assert not a.flags['C_CONTIGUOUS']
        result = my_sum(a)
        assert abs(result - 10.0) < 1e-5


# ==================== SHAPE SPECIALIZATION ====================

class TestShapeSpecialization:
    """Tensor[f32, DYN] compiles a static variant per unique input shape."""

    def test_dyn_syntax_accepted(self):
        """Tensor[f32, DYN] in a type hint does not raise at decoration time."""
        @ml_function
        def my_sum(a: Tensor[f32, DYN]) -> f32:
            return tensor_sum(a)

    def test_dyn_sum_correct(self, backend):
        """DYN function returns correct scalar result."""
        @ml_function
        def my_sum(a: Tensor[f32, DYN]) -> f32:
            return tensor_sum(a)

        a = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        result = my_sum(a)
        assert abs(result - 10.0) < 1e-5

    def test_different_shapes_compiled_separately(self, backend):
        """Calling with shape 4 then shape 8 both produce correct results."""
        @ml_function
        def my_sum(a: Tensor[f32, DYN]) -> f32:
            return tensor_sum(a)

        a4 = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        a8 = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], dtype=np.float32)

        assert abs(my_sum(a4) - 10.0) < 1e-5
        assert abs(my_sum(a8) - 36.0) < 1e-5

    def test_shape_cache_hit(self, backend):
        """Second call with same shape reuses the compiled variant."""
        @ml_function
        def my_sum(a: Tensor[f32, DYN]) -> f32:
            return tensor_sum(a)

        a = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        my_sum(a)
        variants_before = len(my_sum._compiled_variants)
        my_sum(a)
        assert len(my_sum._compiled_variants) == variants_before

    def test_static_and_dyn_mixed(self, backend):
        """Function with one static and one DYN param compiles correctly."""
        @ml_function
        def scale_sum(a: Tensor[f32, 4], b: Tensor[f32, DYN]) -> f32:
            return tensor_sum(tensor_map(a, lambda v: v * 2.0))

        a = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        b = np.array([10.0, 20.0], dtype=np.float32)
        result = scale_sum(a, b)
        assert abs(result - 20.0) < 1e-5

    def test_wrong_ndim_raises(self):
        """Passing a 2D array to Tensor[f32, DYN] (1D) raises ValueError."""
        @ml_function
        def my_sum(a: Tensor[f32, DYN]) -> f32:
            return tensor_sum(a)

        a2d = np.ones((4, 4), dtype=np.float32)
        with pytest.raises(ValueError):
            my_sum(a2d)

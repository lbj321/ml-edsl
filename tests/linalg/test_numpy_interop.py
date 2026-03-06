"""Test numpy interoperability (Phase 8.5a)

Validates that np.ndarray can be passed to @ml_function parameters and that
array outputs are returned as np.ndarray.
"""

import pytest
import numpy as np
from mlir_edsl import ml_function, Array, f32, i32, tensor_map, tensor_sum


# ==================== INPUT INTEROP ====================

class TestNumpyInput:
    """np.ndarray accepted as input, result is ndarray"""

    def test_scalar_return_from_ndarray_input(self, backend):
        """tensor_sum of a numpy array returns correct scalar."""
        @ml_function
        def my_sum(a: Array[f32, 4]) -> f32:
            return tensor_sum(a)

        a = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        result = my_sum(a)
        assert abs(result - 10.0) < 1e-5

    def test_array_return_is_ndarray(self, backend):
        """Array return type comes back as np.ndarray."""
        @ml_function
        def scale(a: Array[f32, 4]) -> Array[f32, 4]:
            return tensor_map(a, lambda v: v * 2.0)

        a = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        result = scale(a)
        assert isinstance(result, np.ndarray)
        np.testing.assert_allclose(result, [2.0, 4.0, 6.0, 8.0], rtol=1e-5)

    def test_list_input_still_works(self, backend):
        """Existing list-based calling convention is unchanged."""
        @ml_function
        def scale(a: Array[f32, 4]) -> Array[f32, 4]:
            return tensor_map(a, lambda v: v * 2.0)

        result = scale([1.0, 2.0, 3.0, 4.0])
        assert isinstance(result, np.ndarray)
        np.testing.assert_allclose(result, [2.0, 4.0, 6.0, 8.0], rtol=1e-5)


# ==================== VALIDATION ====================

class TestNumpyValidation:
    """Wrong dtype or shape raises before calling JIT"""

    def test_wrong_dtype_raises(self):
        """Passing float64 where float32 is expected raises TypeError."""
        @ml_function
        def my_sum(a: Array[f32, 4]) -> f32:
            return tensor_sum(a)

        a = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
        with pytest.raises(TypeError, match="dtype"):
            my_sum(a)

    def test_wrong_shape_raises(self):
        """Passing wrong shape raises ValueError."""
        @ml_function
        def my_sum(a: Array[f32, 4]) -> f32:
            return tensor_sum(a)

        a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        with pytest.raises(ValueError, match="shape"):
            my_sum(a)

    def test_non_contiguous_accepted(self, backend):
        """Non-contiguous ndarray is auto-copied to contiguous before call."""
        @ml_function
        def my_sum(a: Array[f32, 4]) -> f32:
            return tensor_sum(a)

        # Slice every other element from an 8-element array → non-contiguous
        base = np.array([1.0, 0.0, 2.0, 0.0, 3.0, 0.0, 4.0, 0.0], dtype=np.float32)
        a = base[::2]  # [1, 2, 3, 4], non-contiguous
        assert not a.flags['C_CONTIGUOUS']
        result = my_sum(a)
        assert abs(result - 10.0) < 1e-5

"""Test reduction operations execution end-to-end (Phase 8.4)

Validates that reduce, tensor_sum, tensor_max, tensor_min compile through
the full pipeline:
  Python AST → Protobuf → C++ LinalgBuilder → linalg.reduce IR
  → convert-linalg-to-loops → scf.for → LLVM IR → JIT execution
"""

import pytest
import math
import numpy as np
from mlir_edsl import ml_function, Array, f32, i32, reduce, tensor_sum, tensor_max, tensor_min
from mlir_edsl.ast.helpers import to_value


# ==================== tensor_sum ====================

class TestTensorSum:
    """Test tensor_sum: linalg.reduce with addition, init=0.0"""

    def test_sum_positive(self, backend):
        """Sum of positive values."""
        @ml_function
        def my_sum(a: Array[f32, 4]) -> f32:
            return tensor_sum(a)

        assert abs(my_sum(np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)) - 10.0) < 1e-5

    def test_sum_with_negatives(self, backend):
        """Sum including negative values."""
        @ml_function
        def my_sum(a: Array[f32, 4]) -> f32:
            return tensor_sum(a)

        assert abs(my_sum(np.array([-1.0, 2.0, -3.0, 4.0], dtype=np.float32)) - 2.0) < 1e-5

    def test_sum_all_zeros(self, backend):
        """Sum of all zeros is zero."""
        @ml_function
        def my_sum(a: Array[f32, 4]) -> f32:
            return tensor_sum(a)

        assert abs(my_sum(np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32))) < 1e-5

    def test_sum_single_element(self, backend):
        """Sum of a single-element array is the element itself."""
        @ml_function
        def my_sum(a: Array[f32, 1]) -> f32:
            return tensor_sum(a)

        assert abs(my_sum(np.array([42.0], dtype=np.float32)) - 42.0) < 1e-5

    def test_sum_larger_array(self, backend):
        """Sum of 8 elements."""
        @ml_function
        def my_sum(a: Array[f32, 8]) -> f32:
            return tensor_sum(a)

        assert abs(my_sum(np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], dtype=np.float32)) - 36.0) < 1e-4


# ==================== tensor_max ====================

class TestTensorMax:
    """Test tensor_max: linalg.reduce with max(elem, acc), init=-inf"""

    def test_max_positive(self, backend):
        """Max of positive values."""
        @ml_function
        def my_max(a: Array[f32, 4]) -> f32:
            return tensor_max(a)

        assert abs(my_max(np.array([1.0, 4.0, 2.0, 3.0], dtype=np.float32)) - 4.0) < 1e-5

    def test_max_with_negatives(self, backend):
        """Max of mixed positive and negative values."""
        @ml_function
        def my_max(a: Array[f32, 4]) -> f32:
            return tensor_max(a)

        assert abs(my_max(np.array([-1.0, -4.0, -2.0, -3.0], dtype=np.float32)) - (-1.0)) < 1e-5

    def test_max_single_element(self, backend):
        """Max of single-element array is the element."""
        @ml_function
        def my_max(a: Array[f32, 1]) -> f32:
            return tensor_max(a)

        assert abs(my_max(np.array([7.0], dtype=np.float32)) - 7.0) < 1e-5

    def test_max_last_element_is_max(self, backend):
        """Max is the last element in the array."""
        @ml_function
        def my_max(a: Array[f32, 4]) -> f32:
            return tensor_max(a)

        assert abs(my_max(np.array([1.0, 2.0, 3.0, 9.0], dtype=np.float32)) - 9.0) < 1e-5


# ==================== tensor_min ====================

class TestTensorMin:
    """Test tensor_min: linalg.reduce with min(elem, acc), init=+inf"""

    def test_min_positive(self, backend):
        """Min of positive values."""
        @ml_function
        def my_min(a: Array[f32, 4]) -> f32:
            return tensor_min(a)

        assert abs(my_min(np.array([3.0, 1.0, 4.0, 2.0], dtype=np.float32)) - 1.0) < 1e-5

    def test_min_with_negatives(self, backend):
        """Min of mixed values."""
        @ml_function
        def my_min(a: Array[f32, 4]) -> f32:
            return tensor_min(a)

        assert abs(my_min(np.array([1.0, -2.0, 3.0, -4.0], dtype=np.float32)) - (-4.0)) < 1e-5

    def test_min_single_element(self, backend):
        """Min of single-element array is the element."""
        @ml_function
        def my_min(a: Array[f32, 1]) -> f32:
            return tensor_min(a)

        assert abs(my_min(np.array([5.0], dtype=np.float32)) - 5.0) < 1e-5

    def test_min_first_element_is_min(self, backend):
        """Min is the first element in the array."""
        @ml_function
        def my_min(a: Array[f32, 4]) -> f32:
            return tensor_min(a)

        assert abs(my_min(np.array([-9.0, 2.0, 3.0, 4.0], dtype=np.float32)) - (-9.0)) < 1e-5


# ==================== general reduce ====================

class TestReduce:
    """Test the general reduce() op with custom combining functions."""

    def test_custom_sum(self, backend):
        """reduce with addition matches tensor_sum."""
        @ml_function
        def custom_sum(a: Array[f32, 4]) -> f32:
            return reduce(a, to_value(0.0), lambda elem, acc: acc + elem)

        assert abs(custom_sum(np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)) - 10.0) < 1e-5

    def test_custom_product(self, backend):
        """reduce with multiplication computes product."""
        @ml_function
        def product(a: Array[f32, 4]) -> f32:
            return reduce(a, to_value(1.0), lambda elem, acc: acc * elem)

        assert abs(product(np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)) - 24.0) < 1e-4

    def test_reduce_composable_with_other_ops(self, backend):
        """Result of tensor_sum can be used in further arithmetic."""
        @ml_function
        def sum_times_two(a: Array[f32, 4]) -> f32:
            return tensor_sum(a) * to_value(2.0)

        assert abs(sum_times_two(np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)) - 20.0) < 1e-4


# ==================== type validation ====================

class TestReduceTypeValidation:
    """Test that invalid inputs are rejected at Python AST construction time."""

    def test_2d_array_rejected(self):
        """reduce requires a 1D array."""
        from mlir_edsl.ast.nodes.arrays import ArrayLiteral
        from mlir_edsl.types import ArrayType, f32 as f32_type

        arr_2d = ArrayLiteral([[1.0, 2.0], [3.0, 4.0]], ArrayType((2, 2), f32_type))
        with pytest.raises(TypeError, match="1D array"):
            reduce(arr_2d, to_value(0.0), lambda e, a: a + e)

    def test_init_type_mismatch_rejected(self):
        """reduce init must have the same element type as the array."""
        from mlir_edsl.ast.nodes.arrays import ArrayLiteral
        from mlir_edsl.types import ArrayType, f32 as f32_type

        arr = ArrayLiteral([1.0, 2.0, 3.0], ArrayType(3, f32_type))
        with pytest.raises(TypeError, match="init must be a scalar"):
            reduce(arr, to_value(0), lambda e, a: a + e)  # i32 init for f32 array

    def test_body_type_mismatch_rejected(self):
        """reduce body must return the same element type."""
        from mlir_edsl.ast.nodes.arrays import ArrayLiteral
        from mlir_edsl.types import ArrayType, f32 as f32_type

        arr = ArrayLiteral([1.0, 2.0, 3.0], ArrayType(3, f32_type))
        with pytest.raises(TypeError, match="body must return element type"):
            reduce(arr, to_value(0.0), lambda e, a: to_value(1))  # returns i32

"""DYN shape validation tests for linalg.dot and linalg.matmul

When operands have DYN dims the inner-dimension / length compatibility check is
deferred to specialization time (abstract eval with concrete shapes).  These
tests verify that:

  - DYN operands are accepted at decoration time (check deferred)
  - Incompatible concrete shapes raise a clear TypeError at call time
  - Compatible concrete shapes compile and produce correct results
  - Statically incompatible shapes are still caught at decoration time
"""

import numpy as np
import pytest
from mlir_edsl import ml_function, Tensor, f32, i32, dot, matmul
from mlir_edsl.types import DYN


# ==================== DOT — DYN LENGTH VALIDATION ====================

class TestDotDynValidation:
    """linalg.dot length check is deferred for DYN operands."""

    def test_dyn_dot_decoration_accepted(self):
        """Tensor[f32, DYN] dot operands are accepted at decoration time."""
        @ml_function
        def dot_dyn(a: Tensor[f32, DYN], b: Tensor[f32, DYN]) -> f32:
            return dot(a, b)

    def test_dyn_dot_mismatched_lengths_raises_at_call(self, backend):
        """Calling dot with DYN operands of different lengths raises TypeError."""
        @ml_function
        def dot_dyn_mismatch(a: Tensor[f32, DYN], b: Tensor[f32, DYN]) -> f32:
            return dot(a, b)

        a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        b = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        with pytest.raises(TypeError, match="lengths must match"):
            dot_dyn_mismatch(a, b)

    def test_dyn_dot_matching_lengths_correct_result(self, backend):
        """dot with DYN operands of matching length compiles and returns correct result."""
        @ml_function
        def dot_dyn_ok(a: Tensor[f32, DYN], b: Tensor[f32, DYN]) -> f32:
            return dot(a, b)

        a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        b = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        result = dot_dyn_ok(a, b)
        assert abs(result - 14.0) < 1e-5

    def test_static_dot_mismatched_lengths_raises_at_decoration(self):
        """Statically incompatible dot operands are caught at first call."""
        @ml_function
        def dot_static_mismatch(a: Tensor[f32, 3], b: Tensor[f32, 4]) -> f32:
            return dot(a, b)

        with pytest.raises(TypeError, match="lengths must match"):
            dot_static_mismatch(np.ones(3, dtype=np.float32),
                                np.ones(4, dtype=np.float32))


# ==================== MATMUL — DYN INNER DIMENSION VALIDATION ====================

class TestMatmulDynValidation:
    """linalg.matmul inner-dimension check is deferred for DYN operands."""

    def test_dyn_matmul_decoration_accepted(self):
        """Tensor[f32, DYN, DYN] matmul operands are accepted at decoration time."""
        @ml_function
        def matmul_dyn(A: Tensor[f32, DYN, DYN], B: Tensor[f32, DYN, DYN]) -> Tensor[f32, DYN, DYN]:
            return matmul(A, B)

    def test_dyn_matmul_incompatible_inner_dims_raises_at_call(self, backend):
        """Calling matmul with DYN operands whose inner dims mismatch raises TypeError."""
        @ml_function
        def matmul_dyn_mismatch(A: Tensor[f32, DYN, DYN], B: Tensor[f32, DYN, DYN]) -> Tensor[f32, DYN, DYN]:
            return matmul(A, B)

        # A is [2, 3], B is [4, 2] — inner dims 3 != 4
        A = np.ones((2, 3), dtype=np.float32)
        B = np.ones((4, 2), dtype=np.float32)
        with pytest.raises(TypeError, match="inner dimensions must match"):
            matmul_dyn_mismatch(A, B)

    def test_dyn_matmul_compatible_dims_correct_result(self, backend):
        """matmul with DYN operands of compatible dims compiles and returns correct result."""
        @ml_function
        def matmul_dyn_ok(A: Tensor[f32, DYN, DYN], B: Tensor[f32, DYN, DYN]) -> Tensor[f32, DYN, DYN]:
            return matmul(A, B)

        A = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        B = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32)
        result = matmul_dyn_ok(A, B)
        np.testing.assert_allclose(result, [[19.0, 22.0], [43.0, 50.0]], rtol=1e-3)

    def test_static_matmul_incompatible_inner_dims_raises_at_decoration(self):
        """Statically incompatible matmul inner dims are caught at first call."""
        @ml_function
        def matmul_static_mismatch(A: Tensor[f32, 2, 3], B: Tensor[f32, 4, 2]) -> Tensor[f32, 2, 2]:
            return matmul(A, B)

        with pytest.raises(TypeError, match="inner dimensions must match"):
            matmul_static_mismatch(np.ones((2, 3), dtype=np.float32),
                                   np.ones((4, 2), dtype=np.float32))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

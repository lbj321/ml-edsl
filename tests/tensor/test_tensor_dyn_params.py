"""Dynamic tensor input parameter tests (Phase 8.8)

Tensor[f32, DYN] compiles a static variant per unique input shape, identical
to the Array[f32, DYN] shape-specialisation mechanism.
"""

import numpy as np
import pytest
from mlir_edsl import ml_function, Tensor, For, f32, i32
from mlir_edsl.types import DYN


# ==================== SYNTAX / DECORATION ====================

class TestDynTensorSyntax:
    """DYN tensor params must be accepted at decoration time (no backend needed)."""

    def test_dyn_tensor_syntax_accepted(self):
        """Tensor[f32, DYN] in a type hint does not raise at decoration time."""
        @ml_function
        def tdyn_syntax(t: Tensor[f32, DYN]) -> f32:
            return t[0]


# ==================== BASIC EXECUTION ====================

class TestDynTensorExecution:
    """Runtime correctness for Tensor[T, DYN] parameters."""

    def test_dyn_tensor_first_element(self, backend):
        """Access first element of a DYN tensor param."""
        @ml_function
        def tdyn_first(t: Tensor[f32, DYN]) -> f32:
            return t[0]

        result = tdyn_first(np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32))
        assert abs(result - 1.0) < 1e-5

    def test_dyn_tensor_middle_element(self, backend):
        """Access a non-zero index of a DYN tensor param."""
        @ml_function
        def tdyn_mid(t: Tensor[i32, DYN]) -> i32:
            return t[2]

        result = tdyn_mid(np.array([10, 20, 30, 40], dtype=np.int32))
        assert result == 30

    def test_dyn_i32_tensor_first_element(self, backend):
        """DYN i32 tensor — basic access."""
        @ml_function
        def tdyn_i32(t: Tensor[i32, DYN]) -> i32:
            return t[0]

        result = tdyn_i32(np.array([99, 1, 2], dtype=np.int32))
        assert result == 99


# ==================== SHAPE SPECIALISATION ====================

class TestDynTensorShapeSpecialisation:
    """Each unique shape triggers a new compiled variant."""

    def test_two_different_shapes_both_correct(self, backend):
        """Same function called with size-3 then size-6 — both correct."""
        @ml_function
        def tdyn_first_dyn(t: Tensor[f32, DYN]) -> f32:
            return t[0]

        a3 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        a6 = np.array([9.0, 8.0, 7.0, 6.0, 5.0, 4.0], dtype=np.float32)

        assert abs(tdyn_first_dyn(a3) - 1.0) < 1e-5
        assert abs(tdyn_first_dyn(a6) - 9.0) < 1e-5

    def test_shape_cache_hit(self, backend):
        """Second call with same shape reuses compiled variant — no extra compile."""
        @ml_function
        def tdyn_cache(t: Tensor[f32, DYN]) -> f32:
            return t[0]

        a = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        tdyn_cache(a)
        variants_before = len(tdyn_cache._compiled_variants)
        tdyn_cache(a)
        assert len(tdyn_cache._compiled_variants) == variants_before


# ==================== MIXED PARAMS ====================

class TestDynTensorMixedParams:
    """DYN tensor alongside static tensor and scalar params."""

    def test_static_tensor_and_dyn_tensor(self, backend):
        """One static Tensor[f32, 4] and one Tensor[f32, DYN] in same function."""
        @ml_function
        def tdyn_mixed_static(a: Tensor[f32, 4], b: Tensor[f32, DYN]) -> f32:
            return a[0] + b[0]

        a = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        b = np.array([10.0, 20.0], dtype=np.float32)
        result = tdyn_mixed_static(a, b)
        assert abs(result - 11.0) < 1e-5

    def test_dyn_tensor_with_scalar_param(self, backend):
        """DYN tensor param alongside a scalar param."""
        @ml_function
        def tdyn_with_scalar(t: Tensor[f32, DYN], scale: f32) -> f32:
            return t[0] * scale

        t = np.array([3.0, 4.0, 5.0], dtype=np.float32)
        result = tdyn_with_scalar(t, 2.0)
        assert abs(result - 6.0) < 1e-5

    def test_two_dyn_tensor_params(self, backend):
        """Two DYN tensor params — both to_tensor inserts must fire."""
        @ml_function
        def tdyn_two(a: Tensor[f32, DYN], b: Tensor[f32, DYN]) -> f32:
            return a[0] + b[0]

        a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        b = np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float32)
        result = tdyn_two(a, b)
        assert abs(result - 11.0) < 1e-5


# ==================== CONTROL FLOW ====================

class TestDynTensorWithControlFlow:
    """DYN tensor params used inside For loops."""

    def test_dyn_tensor_for_loop_sum(self, backend):
        """Sum all elements of a DYN tensor via For loop."""
        @ml_function
        def tdyn_sum(t: Tensor[f32, DYN], n: i32) -> f32:
            return For(start=0, end=n, init=0.0,
                       body=lambda i, acc: acc + t[i])

        t = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        result = tdyn_sum(t, 4)
        assert abs(result - 10.0) < 1e-5


# ==================== ERROR CASES ====================

class TestDynTensorErrors:
    """Bad inputs to DYN tensor params produce clear errors."""

    def test_dyn_tensor_wrong_ndim_raises(self, backend):
        """Passing a 2D array to Tensor[f32, DYN] (declared 1D) raises."""
        @ml_function
        def tdyn_err_ndim(t: Tensor[f32, DYN]) -> f32:
            return t[0]

        a2d = np.ones((4, 4), dtype=np.float32)
        with pytest.raises((TypeError, ValueError)):
            tdyn_err_ndim(a2d)

    def test_dyn_tensor_wrong_dtype_raises(self, backend):
        """Passing float64 array to Tensor[f32, DYN] raises TypeError."""
        @ml_function
        def tdyn_err_dtype(t: Tensor[f32, DYN]) -> f32:
            return t[0]

        a64 = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        with pytest.raises(TypeError, match="dtype"):
            tdyn_err_dtype(a64)

    def test_dyn_tensor_non_array_raises(self, backend):
        """Passing a Python tuple raises TypeError."""
        @ml_function
        def tdyn_err_type(t: Tensor[f32, DYN]) -> f32:
            return t[0]

        with pytest.raises(TypeError, match="ndarray"):
            tdyn_err_type((1.0, 2.0, 3.0))


# ==================== IR STRUCTURE ====================

class TestDynTensorIR:
    """Verify pre-lowering IR for DYN tensor params."""

    def test_dyn_tensor_specialised_to_static_type(self, check_ir):
        """After shape specialisation the compiled variant has a static tensor type,
        not a dynamic one — DYN is resolved before compilation."""
        @ml_function
        def tdyn_ir(t: Tensor[f32, DYN]) -> f32:
            return t[0]

        tdyn_ir(np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32))

        check_ir("""
        // CHECK: func.func @tdyn_ir
        // CHECK-SAME: memref<4xf32>
        // CHECK: bufferization.to_tensor {{.*}} : memref<4xf32> to tensor<4xf32>
        // CHECK-NOT: tensor<?xf32>
        """)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

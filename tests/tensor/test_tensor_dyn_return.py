"""Dynamic tensor/array return type tests (Phase 8.9)

Functions declared with `-> Tensor[f32, DYN]` or `-> Array[f32, DYN]` use
JAX-style abstract shape evaluation: the concrete input shapes are traced through
the function body to determine the concrete output shape before compilation.
"""

import numpy as np
import pytest
from mlir_edsl import ml_function, Tensor, f32, i32
from mlir_edsl.types import DYN


# ==================== SYNTAX / DECORATION ====================

class TestDynTensorReturnSyntax:
    """DYN tensor return must be accepted at decoration time (no backend needed)."""

    def test_dyn_return_syntax_accepted(self):
        """Tensor[f32, DYN] return hint does not raise at decoration time."""
        @ml_function
        def tr_dyn_syntax(t: Tensor[f32, DYN]) -> Tensor[f32, DYN]:
            return t


# ==================== TENSOR RETURN EXECUTION ====================

class TestDynTensorReturnExecution:
    """Runtime correctness for functions returning Tensor[T, DYN]."""

    def test_tensor_passthrough(self, backend):
        """Tensor[f32, DYN] passthrough: result matches input values."""
        @ml_function
        def tdyn_ret_passthrough(t: Tensor[f32, DYN]) -> Tensor[f32, DYN]:
            return t

        inp = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        result = tdyn_ret_passthrough(inp)
        assert isinstance(result, np.ndarray)
        assert result.shape == (4,)
        np.testing.assert_allclose(result, inp, rtol=1e-5)

    def test_tensor_passthrough_i32(self, backend):
        """Tensor[i32, DYN] passthrough: result matches input values."""
        @ml_function
        def tdyn_ret_i32(t: Tensor[i32, DYN]) -> Tensor[i32, DYN]:
            return t

        inp = np.array([10, 20, 30], dtype=np.int32)
        result = tdyn_ret_i32(inp)
        assert isinstance(result, np.ndarray)
        assert result.shape == (3,)
        np.testing.assert_array_equal(result, inp)

    def test_tensor_passthrough_larger(self, backend):
        """Passthrough of a larger tensor returns all elements correctly."""
        @ml_function
        def tdyn_ret_large(t: Tensor[f32, DYN]) -> Tensor[f32, DYN]:
            return t

        inp = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
        result = tdyn_ret_large(inp)
        assert isinstance(result, np.ndarray)
        assert result.shape == (5,)
        np.testing.assert_allclose(result, inp, rtol=1e-5)


# ==================== SHAPE VARIANTS ====================

class TestDynReturnShapeVariants:
    """Different input sizes compile separate variants with their own concrete return type."""

    def test_two_sizes_each_compile_variant(self, backend):
        """Size-3 and size-6 inputs each trigger a separate compiled variant."""
        @ml_function
        def tdyn_ret_variants(t: Tensor[f32, DYN]) -> Tensor[f32, DYN]:
            return t

        a3 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        a6 = np.array([9.0, 8.0, 7.0, 6.0, 5.0, 4.0], dtype=np.float32)

        r3 = tdyn_ret_variants(a3)
        r6 = tdyn_ret_variants(a6)

        assert r3.shape == (3,)
        assert r6.shape == (6,)
        np.testing.assert_allclose(r3, a3, rtol=1e-5)
        np.testing.assert_allclose(r6, a6, rtol=1e-5)

    def test_variant_count_grows_with_distinct_shapes(self, backend):
        """Each unique shape adds exactly one variant to the cache."""
        @ml_function
        def tdyn_ret_count(t: Tensor[f32, DYN]) -> Tensor[f32, DYN]:
            return t

        tdyn_ret_count(np.ones(4, dtype=np.float32))
        assert len(tdyn_ret_count._compiled_variants) == 1
        tdyn_ret_count(np.ones(8, dtype=np.float32))
        assert len(tdyn_ret_count._compiled_variants) == 2
        # Same size again — no new variant
        tdyn_ret_count(np.ones(4, dtype=np.float32))
        assert len(tdyn_ret_count._compiled_variants) == 2

    def test_same_size_reuses_variant(self, backend):
        """Second call with the same size reuses the compiled variant."""
        @ml_function
        def tdyn_ret_cache(t: Tensor[f32, DYN]) -> Tensor[f32, DYN]:
            return t

        a = np.ones(5, dtype=np.float32)
        tdyn_ret_cache(a)
        before = len(tdyn_ret_cache._compiled_variants)
        tdyn_ret_cache(a)
        assert len(tdyn_ret_cache._compiled_variants) == before


# ==================== IR STRUCTURE ====================

class TestDynReturnIR:
    """The compiled variant IR must have a static tensor type, not `?`."""

    def test_return_type_is_static_in_ir(self, check_ir):
        """After abstract evaluation the return tensor has a concrete static type."""
        @ml_function
        def tdyn_ret_ir(t: Tensor[f32, DYN]) -> Tensor[f32, DYN]:
            return t

        tdyn_ret_ir(np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32))

        check_ir("""
        // CHECK: func.func @tdyn_ret_ir
        // CHECK: tensor<4xf32>
        // CHECK-NOT: tensor<?xf32>
        """)


# ==================== ERROR CASES ====================

class TestDynReturnErrors:
    """Wrong return types produce clear errors at decoration time."""

    def test_static_return_with_dyn_inferred_raises(self):
        """Declaring static return type but body returns DYN tensor raises TypeError."""
        with pytest.raises(TypeError, match="[Rr]eturn type"):
            @ml_function
            def tr_err_static_decl(t: Tensor[f32, DYN]) -> Tensor[f32, 5]:
                # This would return a size-DYN tensor, but the declared type is size-5
                return t

    def test_wrong_element_type_return_raises(self):
        """Element type mismatch in return raises TypeError at decoration."""
        with pytest.raises(TypeError):
            @ml_function
            def tr_err_dtype(t: Tensor[f32, DYN]) -> Tensor[i32, DYN]:
                return t


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

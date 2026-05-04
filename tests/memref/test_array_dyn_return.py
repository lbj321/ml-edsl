"""Dynamic array return type tests (Phase 8.9)

Functions declared with `-> Array[f32, DYN]` use JAX-style abstract shape
evaluation: the concrete input shapes are traced through the function body to
determine the concrete output memref shape before compilation.
"""

import numpy as np
import pytest
from mlir_edsl import ml_function, Array, f32, i32
from mlir_edsl.types import DYN


# ==================== SYNTAX / DECORATION ====================

class TestDynArrayReturnSyntax:
    """DYN array return must be accepted at decoration time (no backend needed)."""

    def test_dyn_return_syntax_accepted(self):
        """Array[f32, DYN] return hint does not raise at decoration time."""
        @ml_function
        def ar_dyn_syntax(a: Array[f32, DYN]) -> Array[f32, DYN]:
            return a


# ==================== EXECUTION ====================

class TestDynArrayReturnExecution:
    """Runtime correctness for functions returning Array[T, DYN]."""

    def test_array_passthrough_f32(self, backend):
        """Array[f32, DYN] passthrough: result matches input values."""
        @ml_function
        def adyn_ret_f32(a: Array[f32, DYN]) -> Array[f32, DYN]:
            return a

        inp = np.array([5.0, 6.0, 7.0], dtype=np.float32)
        result = adyn_ret_f32(inp)
        assert isinstance(result, np.ndarray)
        assert result.shape == (3,)
        np.testing.assert_allclose(result, inp, rtol=1e-5)

    def test_array_passthrough_i32(self, backend):
        """Array[i32, DYN] passthrough exercises the i32 memref output path."""
        @ml_function
        def adyn_ret_i32(a: Array[i32, DYN]) -> Array[i32, DYN]:
            return a

        inp = np.array([10, 20, 30, 40], dtype=np.int32)
        result = adyn_ret_i32(inp)
        assert isinstance(result, np.ndarray)
        assert result.shape == (4,)
        np.testing.assert_array_equal(result, inp)

    def test_array_passthrough_larger(self, backend):
        """Passthrough of a larger array returns all elements correctly."""
        @ml_function
        def adyn_ret_large(a: Array[f32, DYN]) -> Array[f32, DYN]:
            return a

        inp = np.arange(8, dtype=np.float32)
        result = adyn_ret_large(inp)
        assert isinstance(result, np.ndarray)
        assert result.shape == (8,)
        np.testing.assert_allclose(result, inp, rtol=1e-5)


# ==================== SHAPE VARIANTS ====================

class TestDynArrayReturnShapeVariants:
    """Different input sizes compile separate variants with their own concrete return type."""

    def test_two_sizes_each_compile_variant(self, backend):
        """Size-4 and size-7 inputs each trigger a separate compiled variant."""
        @ml_function
        def adyn_ret_variants(a: Array[f32, DYN]) -> Array[f32, DYN]:
            return a

        a4 = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        a7 = np.arange(7, dtype=np.float32)

        r4 = adyn_ret_variants(a4)
        r7 = adyn_ret_variants(a7)

        assert r4.shape == (4,)
        assert r7.shape == (7,)
        np.testing.assert_allclose(r4, a4, rtol=1e-5)
        np.testing.assert_allclose(r7, a7, rtol=1e-5)

    def test_variant_count_grows_with_distinct_shapes(self, backend):
        """Each unique shape adds exactly one variant to the cache."""
        @ml_function
        def adyn_ret_count(a: Array[f32, DYN]) -> Array[f32, DYN]:
            return a

        adyn_ret_count(np.ones(4, dtype=np.float32))
        assert len(adyn_ret_count._compiled_variants) == 1
        adyn_ret_count(np.ones(8, dtype=np.float32))
        assert len(adyn_ret_count._compiled_variants) == 2
        # Same size again — no new variant
        adyn_ret_count(np.ones(4, dtype=np.float32))
        assert len(adyn_ret_count._compiled_variants) == 2

    def test_same_size_reuses_variant(self, backend):
        """Second call with the same size reuses the compiled variant."""
        @ml_function
        def adyn_ret_cache(a: Array[f32, DYN]) -> Array[f32, DYN]:
            return a

        a = np.ones(5, dtype=np.float32)
        adyn_ret_cache(a)
        before = len(adyn_ret_cache._compiled_variants)
        adyn_ret_cache(a)
        assert len(adyn_ret_cache._compiled_variants) == before


# ==================== IR STRUCTURE ====================

class TestDynArrayReturnIR:
    """Compiled array variant IR must have a static memref type, not `?`."""

    def test_return_type_is_static_in_ir(self, check_ir):
        """After abstract evaluation the return memref has a concrete static type."""
        @ml_function
        def adyn_ret_ir(a: Array[f32, DYN]) -> Array[f32, DYN]:
            return a

        adyn_ret_ir(np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32))

        check_ir("""
        // CHECK: func.func @adyn_ret_ir
        // CHECK: memref<5xf32>
        // CHECK-NOT: memref<?xf32>
        """)


# ==================== ERROR CASES ====================

class TestDynArrayReturnErrors:
    """Wrong return types produce clear errors at decoration time."""

    def test_static_return_with_dyn_inferred_raises(self):
        """Declaring static return type but body returns DYN array raises TypeError."""
        @ml_function
        def ar_err_static_decl(a: Array[f32, DYN]) -> Array[f32, 5]:
            return a

        with pytest.raises(TypeError, match="[Rr]eturn type"):
            ar_err_static_decl(np.ones(4, dtype=np.float32))

    def test_wrong_element_type_return_raises(self):
        """Element type mismatch in return raises TypeError at first call."""
        @ml_function
        def ar_err_dtype(a: Array[f32, DYN]) -> Array[i32, DYN]:
            return a

        with pytest.raises(TypeError):
            ar_err_dtype(np.ones(4, dtype=np.float32))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

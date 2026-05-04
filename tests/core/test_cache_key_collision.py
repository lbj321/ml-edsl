"""Regression tests for compiled function cache key collisions.

Two distinct bugs reproduced here:

1. Static name collision: two @ml_function wrappers sharing the same Python
   function name (e.g. both named "fn") compile to the same backend key.
   The second wrapper finds has_function("fn") = True, skips compilation, but
   backend._signatures["fn"] still holds the FIRST function's parameter types.
   execute_function builds a wrong ctypes descriptor and the JIT code
   (hardcoded for size 4) runs on a size-8 input → returns 4.0 instead of 8.0.

2. DYN auto-invalidate crash: after my_sum(a4) finalizes the JIT (lowering
   the module to LLVM dialect), my_sum(a8) calls compileFunction which clears
   the executor but leaves the module in LLVM dialect. Adding a new linalg
   function to a lowered module corrupts the IR → segfault.
"""
import numpy as np
import pytest

from mlir_edsl import ml_function, Tensor, f32, tensor_sum
from mlir_edsl.types import DYN


# Two factories that produce @ml_function wrappers with the same Python
# function name ("fn") but different static shapes.
def _make_fn_size4():
    @ml_function
    def fn(a: Tensor[f32, 4]) -> f32:
        return tensor_sum(a)
    return fn


def _make_fn_size8():
    @ml_function
    def fn(a: Tensor[f32, 8]) -> f32:
        return tensor_sum(a)
    return fn


class TestCacheKeyCollision:

    def test_static_name_collision(self, backend):
        """Two wrappers for 'fn' with different shapes: the second must compile
        its own backend function, not reuse the first's compiled code.

        Bug path: compile_function sees has_function("fn") == True after fn4
        compiles, skips compilation for fn8, but _signatures["fn"] retains
        Tensor[f32,4] types. The JIT loop for size-4 executes on an 8-element
        array and returns 4.0 instead of 8.0.
        """
        fn4 = _make_fn_size4()
        fn8 = _make_fn_size8()

        a4 = np.ones(4, dtype=np.float32)
        a8 = np.ones(8, dtype=np.float32)

        r4 = fn4(a4)
        assert abs(r4 - 4.0) < 1e-5, f"fn4: expected 4.0, got {r4}"

        r8 = fn8(a8)
        assert abs(r8 - 8.0) < 1e-5, (
            f"fn8: expected 8.0, got {r8} — cache collision: "
            "fn4's compiled code ran on fn8's input"
        )

    def test_dyn_two_shapes_auto_invalidate(self, backend):
        """DYN function called with shape 4 then shape 8 within the same test.

        After my_sum(a4) executes, the module is lowered to LLVM dialect and
        the JIT is finalized. my_sum(a8) then calls compileFunction which clears
        the executor (auto-invalidate) but leaves the module in LLVM dialect.
        Adding a new linalg function to a lowered LLVM module corrupts the IR
        and may cause a segfault.
        """
        @ml_function
        def my_sum(a: Tensor[f32, DYN]) -> f32:
            return tensor_sum(a)

        a4 = np.ones(4, dtype=np.float32)
        a8 = np.ones(8, dtype=np.float32)

        r4 = my_sum(a4)
        assert abs(r4 - 4.0) < 1e-5, f"shape-4: expected 4.0, got {r4}"

        # Second shape triggers auto-invalidate; may segfault before fix
        r8 = my_sum(a8)
        assert abs(r8 - 8.0) < 1e-5, f"shape-8: expected 8.0, got {r8}"

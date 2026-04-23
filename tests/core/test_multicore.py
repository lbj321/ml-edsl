"""Test CPU multicore execution via async dialect parallelism

Validates that the async parallel-for pipeline:
  linalg.matmul → scf.forall (64x64 tiles) → scf.parallel → async.execute
  → libmlir_async_runtime.so thread pool

produces correct numerical results across matrix sizes that exercise
the parallel tiling path (multiples of the 64x64 outer tile size).
"""

import pytest
import numpy as np
from mlir_edsl import ml_function, Tensor, f32


class TestMulticoreCorrectness:
    """Verify that async-parallelized matmul produces correct results.

    These are the primary correctness tests. If async introduces data races
    or incorrect tiling the numbers will be wrong.
    """

    def test_matmul_32x32_ones(self, backend):
        """32x32: ones @ ones == 32*ones (2x2 = 4 parallel tiles)"""
        @ml_function
        def mm(A: Tensor[f32, 32, 32], B: Tensor[f32, 32, 32]) -> Tensor[f32, 32, 32]:
            return A @ B

        A = np.ones((32, 32), dtype=np.float32)
        B = np.ones((32, 32), dtype=np.float32)
        result = mm(A, B)

        np.testing.assert_allclose(result, np.full((32, 32), 32.0), atol=1e-4)

    def test_matmul_64x64_ones(self, backend):
        """64x64: ones @ ones == 64*ones (4x4 = 16 parallel tiles)"""
        @ml_function
        def mm(A: Tensor[f32, 64, 64], B: Tensor[f32, 64, 64]) -> Tensor[f32, 64, 64]:
            return A @ B

        A = np.ones((64, 64), dtype=np.float32)
        B = np.ones((64, 64), dtype=np.float32)
        result = mm(A, B)

        np.testing.assert_allclose(result, np.full((64, 64), 64.0), atol=1e-4)

    def test_matmul_32x32_vs_numpy(self, backend):
        """32x32 random matmul matches numpy reference"""
        @ml_function
        def mm(A: Tensor[f32, 32, 32], B: Tensor[f32, 32, 32]) -> Tensor[f32, 32, 32]:
            return A @ B

        rng = np.random.default_rng(0)
        A = rng.random((32, 32)).astype(np.float32)
        B = rng.random((32, 32)).astype(np.float32)
        result = mm(A, B)

        np.testing.assert_allclose(result, A @ B, rtol=1e-4, atol=1e-4)

    def test_matmul_64x64_vs_numpy(self, backend):
        """64x64 random matmul matches numpy reference"""
        @ml_function
        def mm(A: Tensor[f32, 64, 64], B: Tensor[f32, 64, 64]) -> Tensor[f32, 64, 64]:
            return A @ B

        rng = np.random.default_rng(1)
        A = rng.random((64, 64)).astype(np.float32)
        B = rng.random((64, 64)).astype(np.float32)
        result = mm(A, B)

        np.testing.assert_allclose(result, A @ B, rtol=1e-4, atol=1e-4)

    def test_matmul_identity_32x32(self, backend):
        """A @ I == A for 32x32"""
        @ml_function
        def mm(A: Tensor[f32, 32, 32], B: Tensor[f32, 32, 32]) -> Tensor[f32, 32, 32]:
            return A @ B

        rng = np.random.default_rng(2)
        A = rng.random((32, 32)).astype(np.float32)
        I = np.eye(32, dtype=np.float32)
        result = mm(A, I)

        np.testing.assert_allclose(result, A, rtol=1e-4, atol=1e-5)

    def test_matmul_repeated_calls(self, backend):
        """Multiple calls to the same compiled function produce consistent results"""
        @ml_function
        def mm(A: Tensor[f32, 32, 32], B: Tensor[f32, 32, 32]) -> Tensor[f32, 32, 32]:
            return A @ B

        rng = np.random.default_rng(3)
        A = rng.random((32, 32)).astype(np.float32)
        B = rng.random((32, 32)).astype(np.float32)
        expected = A @ B

        for _ in range(5):
            result = mm(A, B)
            np.testing.assert_allclose(result, expected, rtol=1e-4, atol=1e-4)


class TestMulticoreIR:
    """Verify that the async runtime is actually wired into the lowered IR.

    Checks that the pipeline emits mlirAsyncRuntime calls — if these are
    absent, we are running single-threaded regardless of correctness.
    """

    # KNOWN ISSUE: mlir_async_runtime owns a process-global thread pool that
    # does not reset when the JIT clears between tests (clean_module fixture).
    # After 6 async executions each followed by clear(), the pool is in an
    # inconsistent state and the 7th execution (this test) may segfault on the
    # first pytest run. Subsequent runs pass because the pool is already warm.
    #
    # Fix options:
    #   A) Session-scoped async warmup fixture that runs one dummy async call
    #      before any tests, so the pool is stable before clean_module cycles start.
    #   B) Don't clear the JIT when async is active (teach clear() to only
    #      reset the function table, not tear down the JIT entirely).
    #   C) pytest-forked: isolate each test in its own subprocess.
    @pytest.mark.xfail(
        strict=False,
        reason="async runtime thread pool lifecycle: may segfault on first run after repeated JIT clear()"
    )
    def test_async_runtime_calls_present(self, backend, check_lowered_ir):
        """Lowered IR must contain async runtime symbols including barrier ops"""
        @ml_function
        def mm(A: Tensor[f32, 64, 64], B: Tensor[f32, 64, 64]) -> Tensor[f32, 64, 64]:
            return A @ B

        mm(np.ones((64, 64), dtype=np.float32), np.ones((64, 64), dtype=np.float32))

        check_lowered_ir("""
        // CHECK: mlirAsyncRuntimeCreateGroup
        // CHECK: mlirAsyncRuntimeAwaitAllInGroup
        """, after="convert-async-to-llvm")

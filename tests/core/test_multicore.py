"""CPU multicore matmul correctness tests.

LinalgMatmulParallelTilingPass tiles linalg.matmul to 64x64 scf.forall outer
blocks, converted to omp.parallel via forall-to-parallel + convert-scf-to-openmp.
The inner LinalgMatmulTilingPass then tiles each block to 8x8 for vectorization.

Sizes < 64: no parallel tiling fires (matrix smaller than one tile).
Sizes >= 64: at least one outer forall iteration; true parallelism.
Non-multiples of 64: boundary tiles handled by fallback scf.for.
"""

import numpy as np
import pytest

from mlir_edsl import ml_function, Tensor, f32


class TestMulticoreMatmul:
    """Correctness tests for matmul using the OpenMP parallel tiling pipeline."""

    def test_matmul_8x8(self, backend):
        """Below tiling threshold — no forall fires, serial path only."""
        @ml_function
        def matmul(A: Tensor[f32, 8, 8], B: Tensor[f32, 8, 8]) -> Tensor[f32, 8, 8]:
            return A @ B

        A = np.random.rand(8, 8).astype(np.float32)
        B = np.random.rand(8, 8).astype(np.float32)
        np.testing.assert_allclose(matmul(A, B), A @ B, rtol=1e-4, atol=1e-4)

    def test_matmul_64x64(self, backend):
        """Exactly one 64x64 tile — minimal parallel case."""
        @ml_function
        def matmul(A: Tensor[f32, 64, 64], B: Tensor[f32, 64, 64]) -> Tensor[f32, 64, 64]:
            return A @ B

        A = np.random.rand(64, 64).astype(np.float32)
        B = np.random.rand(64, 64).astype(np.float32)
        np.testing.assert_allclose(matmul(A, B), A @ B, rtol=1e-4, atol=1e-4)

    def test_matmul_128x128(self, backend):
        """2x2 tile grid — four parallel tiles."""
        @ml_function
        def matmul(A: Tensor[f32, 128, 128], B: Tensor[f32, 128, 128]) -> Tensor[f32, 128, 128]:
            return A @ B

        A = np.random.rand(128, 128).astype(np.float32)
        B = np.random.rand(128, 128).astype(np.float32)
        np.testing.assert_allclose(matmul(A, B), A @ B, rtol=1e-4, atol=1e-4)

    def test_matmul_96x96(self, backend):
        """Non-multiple of 64 — boundary tile (32-wide) exercises fallback path."""
        @ml_function
        def matmul(A: Tensor[f32, 96, 96], B: Tensor[f32, 96, 96]) -> Tensor[f32, 96, 96]:
            return A @ B

        A = np.random.rand(96, 96).astype(np.float32)
        B = np.random.rand(96, 96).astype(np.float32)
        np.testing.assert_allclose(matmul(A, B), A @ B, rtol=1e-4, atol=1e-4)

    def test_matmul_256x256(self, backend):
        """4x4 tile grid — stresses multi-thread scheduling."""
        @ml_function
        def matmul(A: Tensor[f32, 256, 256], B: Tensor[f32, 256, 256]) -> Tensor[f32, 256, 256]:
            return A @ B

        A = np.random.rand(256, 256).astype(np.float32)
        B = np.random.rand(256, 256).astype(np.float32)
        np.testing.assert_allclose(matmul(A, B), A @ B, rtol=1e-3, atol=1e-3)

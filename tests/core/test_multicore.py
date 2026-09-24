"""CPU multicore matmul and dense layer correctness tests.

Matmuls chooseStrategy accepts go through the blocked matmul passes (an
ic x jc scf.forall over a BLIS loop nest); the rest take the 64x64 forall +
8x8 fallback tiling. scf.forall is converted to omp.parallel via
forall-to-parallel + convert-scf-to-openmp. Dense-layer epilogues (bias,
relu) run unfused after the matmul while CPU epilogue fusion is disabled.
"""

import numpy as np

from mlir_edsl import ml_function, Tensor, f32, relu


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


class TestMulticoreDenseLayer:
    """Correctness tests for matmul+bias+relu: a blocked matmul followed by an
    unfused epilogue."""

    def test_dense_relu_64x64(self, backend):
        """64x64 dense layer with relu."""
        @ml_function
        def dense(X: Tensor[f32, 64, 64], W: Tensor[f32, 64, 64], b: Tensor[f32, 64]) -> Tensor[f32, 64, 64]:
            return relu(X @ W + b)

        X = np.random.rand(64, 64).astype(np.float32)
        W = np.random.rand(64, 64).astype(np.float32)
        b = np.random.rand(64).astype(np.float32)
        np.testing.assert_allclose(dense(X, W, b), np.maximum(X @ W + b, 0.0), rtol=1e-3, atol=1e-3)

    def test_dense_relu_128x128(self, backend):
        """128x128 dense layer with relu."""
        @ml_function
        def dense(X: Tensor[f32, 128, 128], W: Tensor[f32, 128, 128], b: Tensor[f32, 128]) -> Tensor[f32, 128, 128]:
            return relu(X @ W + b)

        X = np.random.rand(128, 128).astype(np.float32)
        W = np.random.rand(128, 128).astype(np.float32)
        b = np.random.rand(128).astype(np.float32)
        np.testing.assert_allclose(dense(X, W, b), np.maximum(X @ W + b, 0.0), rtol=1e-3, atol=1e-3)

    def test_dense_relu_non_multiple(self, backend):
        """96x96 dense layer with relu (MC = NC = 96)."""
        @ml_function
        def dense(X: Tensor[f32, 96, 96], W: Tensor[f32, 96, 96], b: Tensor[f32, 96]) -> Tensor[f32, 96, 96]:
            return relu(X @ W + b)

        X = np.random.rand(96, 96).astype(np.float32)
        W = np.random.rand(96, 96).astype(np.float32)
        b = np.random.rand(96).astype(np.float32)
        np.testing.assert_allclose(dense(X, W, b), np.maximum(X @ W + b, 0.0), rtol=1e-3, atol=1e-3)

    def test_dense_no_relu_64x64(self, backend):
        """Bias-only (no relu) 64x64 dense layer."""
        @ml_function
        def dense(X: Tensor[f32, 64, 64], W: Tensor[f32, 64, 64], b: Tensor[f32, 64]) -> Tensor[f32, 64, 64]:
            return X @ W + b

        X = np.random.rand(64, 64).astype(np.float32)
        W = np.random.rand(64, 64).astype(np.float32)
        b = np.random.rand(64).astype(np.float32)
        np.testing.assert_allclose(dense(X, W, b), X @ W + b, rtol=1e-3, atol=1e-3)

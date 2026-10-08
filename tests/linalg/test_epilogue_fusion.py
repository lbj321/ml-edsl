"""Tests for matmul -> bias_add -> relu chains on CPU.

The matmul goes through the blocked matmul passes, which fuse bias_add/relu
into its ic x jc forall. These tests check that chains compute the right
values; the IR structure is covered in test_blocked_matmul_ir.py.
"""

import numpy as np
from mlir_edsl import ml_function, Tensor, f32, matmul, relu


# ==================== RUNTIME EXECUTION ====================

class TestEpilogueFusionExecution:
    """Correctness of matmul -> bias_add -> relu chains, tile-aligned and not."""

    def test_matmul_bias_relu_tile_aligned(self, backend):
        """128x128 is an exact multiple of the 64x64 outer tile."""
        @ml_function
        def dense(A: Tensor[f32, 128, 128], B: Tensor[f32, 128, 128],
                  b: Tensor[f32, 128]) -> Tensor[f32, 128, 128]:
            return relu(matmul(A, B) + b)

        rng = np.random.default_rng(0)
        A = rng.standard_normal((128, 128)).astype(np.float32)
        B = rng.standard_normal((128, 128)).astype(np.float32)
        bias = rng.standard_normal(128).astype(np.float32)

        result = dense(A, B, bias)
        expected = np.maximum(A @ B + bias, 0)
        np.testing.assert_allclose(result, expected, rtol=1e-3, atol=1e-3)

    def test_matmul_bias_relu_boundary_tile(self, backend):
        """96x96 leaves a 32-wide remainder tile at the 64x64 outer tiling."""
        @ml_function
        def dense(A: Tensor[f32, 96, 96], B: Tensor[f32, 96, 96],
                  b: Tensor[f32, 96]) -> Tensor[f32, 96, 96]:
            return relu(matmul(A, B) + b)

        rng = np.random.default_rng(1)
        A = rng.standard_normal((96, 96)).astype(np.float32)
        B = rng.standard_normal((96, 96)).astype(np.float32)
        bias = rng.standard_normal(96).astype(np.float32)

        result = dense(A, B, bias)
        expected = np.maximum(A @ B + bias, 0)
        np.testing.assert_allclose(result, expected, rtol=1e-3, atol=1e-3)

    def test_matmul_bias_relu_smaller_than_tile(self, backend):
        """32x32 is smaller than the 64x64 outer tile: a single partial tile."""
        @ml_function
        def dense(A: Tensor[f32, 32, 32], B: Tensor[f32, 32, 32],
                  b: Tensor[f32, 32]) -> Tensor[f32, 32, 32]:
            return relu(matmul(A, B) + b)

        rng = np.random.default_rng(2)
        A = rng.standard_normal((32, 32)).astype(np.float32)
        B = rng.standard_normal((32, 32)).astype(np.float32)
        bias = rng.standard_normal(32).astype(np.float32)

        result = dense(A, B, bias)
        expected = np.maximum(A @ B + bias, 0)
        np.testing.assert_allclose(result, expected, rtol=1e-3, atol=1e-3)

    def test_relu_actually_clamps_negatives(self, backend):
        """Sanity check that relu is applied, not just bias_add: an
        all-negative pre-activation must come out as all zeros."""
        @ml_function
        def dense(A: Tensor[f32, 64, 64], B: Tensor[f32, 64, 64],
                  b: Tensor[f32, 64]) -> Tensor[f32, 64, 64]:
            return relu(matmul(A, B) + b)

        A = np.zeros((64, 64), dtype=np.float32)
        B = np.zeros((64, 64), dtype=np.float32)
        bias = np.full(64, -1.0, dtype=np.float32)

        result = dense(A, B, bias)
        np.testing.assert_allclose(result, np.zeros((64, 64)), atol=1e-6)


class TestFallbackMatmulTilingExecution:
    """Correctness of matmuls with no relu epilogue."""

    def test_bare_matmul_no_epilogue(self, backend):
        """128x128 matmul alone (no bias/relu) still tiles and executes correctly."""
        @ml_function
        def mm_fn(A: Tensor[f32, 128, 128], B: Tensor[f32, 128, 128]) -> Tensor[f32, 128, 128]:
            return matmul(A, B)

        rng = np.random.default_rng(3)
        A = rng.standard_normal((128, 128)).astype(np.float32)
        B = rng.standard_normal((128, 128)).astype(np.float32)

        result = mm_fn(A, B)
        np.testing.assert_allclose(result, A @ B, rtol=1e-3, atol=1e-3)

    def test_matmul_bias_without_relu(self, backend):
        """bias_add with no relu: the matmul is blocked and bias_add is
        tiled separately further down the pipeline."""
        @ml_function
        def biased(A: Tensor[f32, 128, 128], B: Tensor[f32, 128, 128],
                   b: Tensor[f32, 128]) -> Tensor[f32, 128, 128]:
            return matmul(A, B) + b

        rng = np.random.default_rng(4)
        A = rng.standard_normal((128, 128)).astype(np.float32)
        B = rng.standard_normal((128, 128)).astype(np.float32)
        bias = rng.standard_normal(128).astype(np.float32)

        result = biased(A, B, bias)
        expected = A @ B + bias
        np.testing.assert_allclose(result, expected, rtol=1e-3, atol=1e-3)


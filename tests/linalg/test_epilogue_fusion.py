"""Tests for matmul -> bias_add -> relu chains on CPU.

Epilogue fusion is disabled on CPU for now: LinalgOuterTileAndFusePass is
commented out of buildCPUPipeline, so the matmul goes through the blocked
matmul passes and bias_add/relu run as their own ops afterwards. The
execution tests check that chains still compute the right values. The IR
tests describe the old fused 64x64 structure and stay skipped until
INTEGRATION.md Step 4 fuses the epilogue into the blocked accumulator.
"""

import numpy as np
import pytest
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


# ==================== POST-LOWERING IR ====================

class TestEpilogueFusionIR:
    """IR structure after linalg-outer-tile-and-fuse: verifies the fused
    chain lands inside one scf.forall and that K stays untiled."""

    @pytest.mark.skip(
        reason=(
            "epilogue fusion is disabled on CPU: LinalgOuterTileAndFusePass "
            "is commented out of buildCPUPipeline, and dense layers go "
            "through the blocked matmul passes with the epilogue unfused. "
            "Revisit with INTEGRATION.md Step 4. "
        )
    )
    def test_relu_epilogue_fused_into_one_forall(self, check_lowered_ir):
        @ml_function
        def dense(A: Tensor[f32, 128, 128], B: Tensor[f32, 128, 128],
                  b: Tensor[f32, 128]) -> Tensor[f32, 128, 128]:
            return relu(matmul(A, B) + b)

        dense(np.ones((128, 128), dtype=np.float32),
              np.ones((128, 128), dtype=np.float32),
              np.ones(128, dtype=np.float32))
        check_lowered_ir("""
        // CHECK: scf.forall (
        // CHECK: linalg.fill
        // CHECK: linalg.matmul
        // CHECK: library_call = "bias_add"
        // CHECK: library_call = "relu"
        // CHECK: scf.forall.in_parallel
        """, after="linalg-outer-tile-and-fuse")

    @pytest.mark.skip(
        reason=(
            "epilogue fusion is disabled on CPU: LinalgOuterTileAndFusePass "
            "is commented out of buildCPUPipeline, and dense layers go "
            "through the blocked matmul passes with the epilogue unfused. "
            "Revisit with INTEGRATION.md Step 4. "
        )
    )
    def test_outer_tile_size_is_64(self, check_lowered_ir):
        @ml_function
        def dense(A: Tensor[f32, 128, 128], B: Tensor[f32, 128, 128],
                  b: Tensor[f32, 128]) -> Tensor[f32, 128, 128]:
            return relu(matmul(A, B) + b)

        dense(np.ones((128, 128), dtype=np.float32),
              np.ones((128, 128), dtype=np.float32),
              np.ones(128, dtype=np.float32))
        check_lowered_ir("""
        // CHECK: scf.forall {{.*}} step (64, 64)
        """, after="linalg-outer-tile-and-fuse")

    @pytest.mark.skip(
        reason=(
            "epilogue fusion is disabled on CPU: LinalgOuterTileAndFusePass "
            "is commented out of buildCPUPipeline, and dense layers go "
            "through the blocked matmul passes with the epilogue unfused. "
            "Revisit with INTEGRATION.md Step 4. "
        )
    )
    def test_fused_matmul_keeps_full_k(self, check_lowered_ir):
        """The fused matmul is sliced to 64 wide on M/N but keeps the full
        128-wide K dimension: relu has no K dimension to slice against, so
        producer fusion never tiles K. Checked after the canonicalizer that
        immediately follows linalg-outer-tile-and-fuse, which removes the
        dead top-level (pre-fusion) matmul copy so there's only one match."""
        @ml_function
        def dense(A: Tensor[f32, 128, 128], B: Tensor[f32, 128, 128],
                  b: Tensor[f32, 128]) -> Tensor[f32, 128, 128]:
            return relu(matmul(A, B) + b)

        dense(np.ones((128, 128), dtype=np.float32),
              np.ones((128, 128), dtype=np.float32),
              np.ones(128, dtype=np.float32))
        check_lowered_ir("""
        // CHECK: linalg.matmul
        // CHECK-SAME: tensor<64x128xf32>
        // CHECK-SAME: tensor<128x64xf32>
        """, after="canonicalize")

    @pytest.mark.skip(
        reason=(
            "Asserts the pre-blocked-path tiling structure. Bare matmuls "
            "now go through the blocked matmul passes (4x16 register tile, "
            "serial loop nest), so there is no 64x64 scf.forall, no 8x8 "
            "extract_slice and no omp.parallel to find. Needs rewriting "
            "against the blocked path rather than un-skipping. "
        )
    )
    def test_bare_matmul_fused(self, check_lowered_ir):
        """With no relu epilogue, LinalgOuterTileAndFusePass uses the bare
        matmul itself as the fusion root, pulling its fill producer into
        the same scf.forall."""
        @ml_function
        def mm_fn(A: Tensor[f32, 128, 128], B: Tensor[f32, 128, 128]) -> Tensor[f32, 128, 128]:
            return matmul(A, B)

        mm_fn(np.ones((128, 128), dtype=np.float32),
              np.ones((128, 128), dtype=np.float32))
        check_lowered_ir("""
        // CHECK: scf.forall (
        // CHECK: linalg.fill
        // CHECK: linalg.matmul
        // CHECK: scf.forall.in_parallel
        """, after="linalg-outer-tile-and-fuse")


class TestFallbackMatmulTilingIR:
    """IR structure after linalg-tile-matmul-forall: verifies the fallback
    pass tiles matmuls the fusion pass didn't touch, and skips ones it did."""

    @pytest.mark.skip(
        reason=(
            "epilogue fusion is disabled on CPU: LinalgOuterTileAndFusePass "
            "is commented out of buildCPUPipeline, and dense layers go "
            "through the blocked matmul passes with the epilogue unfused. "
            "Revisit with INTEGRATION.md Step 4. "
        )
    )
    def test_fused_matmul_not_retiled(self, check_lowered_ir):
        """A matmul already fused into an scf.forall must not get a second,
        redundant outer-tiling forall wrapped around it."""
        @ml_function
        def dense(A: Tensor[f32, 128, 128], B: Tensor[f32, 128, 128],
                  b: Tensor[f32, 128]) -> Tensor[f32, 128, 128]:
            return relu(matmul(A, B) + b)

        dense(np.ones((128, 128), dtype=np.float32),
              np.ones((128, 128), dtype=np.float32),
              np.ones(128, dtype=np.float32))
        check_lowered_ir("""
        // CHECK: scf.forall (
        // CHECK-NOT: scf.forall (
        """, after="linalg-tile-matmul-forall")

    @pytest.mark.skip(
        reason=(
            "Asserts the pre-blocked-path tiling structure. Bare matmuls "
            "now go through the blocked matmul passes (4x16 register tile, "
            "serial loop nest), so there is no 64x64 scf.forall, no 8x8 "
            "extract_slice and no omp.parallel to find. Needs rewriting "
            "against the blocked path rather than un-skipping. "
        )
    )
    def test_bare_matmul_not_retiled(self, check_lowered_ir):
        """A matmul with no epilogue is already fused into an scf.forall by
        LinalgOuterTileAndFusePass, so the fallback pass must not wrap it
        in a second, redundant outer-tiling forall."""
        @ml_function
        def mm_fn(A: Tensor[f32, 128, 128], B: Tensor[f32, 128, 128]) -> Tensor[f32, 128, 128]:
            return matmul(A, B)

        mm_fn(np.ones((128, 128), dtype=np.float32),
              np.ones((128, 128), dtype=np.float32))
        check_lowered_ir("""
        // CHECK: scf.forall (
        // CHECK-NOT: scf.forall (
        """, after="linalg-tile-matmul-forall")

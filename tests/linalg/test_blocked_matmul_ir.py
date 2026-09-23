"""IR structure tests for the blocked matmul passes (INTEGRATION.md Steps 1-2).

A bare f32 matmul whose shape chooseStrategy accepts is distributed over an
ic x jc forall, tiled into a BLIS loop nest over a 4x16 register tile with A
and B packed into contiguous panels, and k-tiled into the register kernel —
one pass per stage. Shapes the guard rejects must keep the old 64x64 / 8x8x8
structure instead.
"""

import numpy as np
from mlir_edsl import ml_function, Tensor, f32, matmul

DISTRIBUTE = "linalg-matmul-blocked-distribute"
TILE_AND_PACK = "linalg-matmul-blocked-tile-and-pack"
KERNEL = "linalg-matmul-blocked-kernel"

# 256 is accepted by chooseStrategy: f32, static, square, power of two, no
# linalg consumer. Large enough for the forall, jr and ir to survive; pc has
# one iteration (K == KC).
N = 256

# 128 gives MC = NC = 128, so the forall has a single iteration and
# canonicalize removes it between the distribute and tile-and-pack passes.
N_SINGLE_TILE = 128

# 24 fails the cache-block search (no MC multiple of MR=4 divides it that also
# satisfies the NR=16 column block), so it falls back to the old path.
N_REJECTED = 24


def _run(n):
    """Compile and call an NxN bare matmul, returning nothing."""
    @ml_function
    def mm_fn(A: Tensor[f32, n, n], B: Tensor[f32, n, n]) -> Tensor[f32, n, n]:
        return matmul(A, B)

    mm_fn(np.ones((n, n), dtype=np.float32), np.ones((n, n), dtype=np.float32))


class TestBlockedMatmulStructure:
    """The loop nest and packed panels the blocked passes produce."""

    def test_produces_two_dimensional_forall(self, check_lowered_ir):
        """ic and jc are fused into one forall over MC x NC tiles of C."""
        _run(N)
        check_lowered_ir("""
        // CHECK: scf.forall ({{.*}}, {{.*}}) = (0, 0) to (256, 256) step (128, 256)
        """, after=DISTRIBUTE)

    def test_register_tile_is_marked_and_4x16x1(self, check_lowered_ir):
        """The innermost tile is MR x NR x 1 and records its strategy."""
        _run(N)
        check_lowered_ir("""
        // CHECK: linalg.matmul {mlir_edsl.blocked = {kc = 256 : i64, mc = 128 : i64, mr = 4 : i64, nc = 256 : i64, nr = 16 : i64, pack_a = true, pack_b = true, stage = "kernel", vectorize = true}}
        // CHECK-SAME: tensor<4x1xf32>, tensor<1x16xf32>
        // CHECK-SAME: tensor<4x16xf32>
        """, after=KERNEL)

    def test_packs_b_into_contiguous_panel(self, check_lowered_ir):
        """B~ is (NC/NR) x KC x NR, built by a vectorized copy loop."""
        _run(N)
        check_lowered_ir("""
        // CHECK: vector.transfer_read
        // CHECK: vector.transfer_write {{.*}} vector<8x16xf32>
        // CHECK-SAME: tensor<256x16xf32>
        """, after=TILE_AND_PACK)

    def test_packs_a_into_k_major_panel(self, check_lowered_ir):
        """A~ is (MC/MR) x KC x MR, built by a tiled [1,0] transpose."""
        _run(N)
        check_lowered_ir("""
        // CHECK: linalg.transpose ins({{.*}} : tensor<4x8xf32>)
        // CHECK-SAME: outs({{.*}} : tensor<8x4xf32>)
        // CHECK-SAME: permutation = [1, 0]
        """, after=TILE_AND_PACK)

    def test_a_untranspose_is_fused_into_k_loop(self, check_lowered_ir):
        """The un-transpose hoisting leaves behind becomes a per-k-step 1xMR."""
        _run(N)
        check_lowered_ir("""
        // CHECK: linalg.transpose ins({{.*}} : tensor<1x4xf32>)
        // CHECK-SAME: outs({{.*}} : tensor<4x1xf32>)
        """, after=KERNEL)

    def test_transposing_transfer_is_lowered(self, check_lowered_ir):
        """No permuting transfer survives to reach convert-vector-to-llvm.

        A transposing transfer_read lowers to a vinsertps chain; shuffles are
        worth ~11% on the packed path.
        """
        _run(N)
        check_lowered_ir("""
        // CHECK-NOT: permutation_map
        """, after="vector-transpose-lowering")

    def test_microkernel_is_outerproduct(self, check_lowered_ir):
        """The reused pipeline passes turn the register tile into FMAs."""
        _run(N)
        check_lowered_ir("""
        // CHECK: vector.outerproduct
        // CHECK-NOT: linalg.matmul
        """, after="vector-contract-to-outerproduct")


class TestBlockedMatmulStages:
    """Each pass picks up the tiles the previous one left, by stage."""

    def test_tile_and_pack_advances_every_distributed_tile(
            self, check_lowered_ir):
        """No tile is left at the distributed stage after tile-and-pack."""
        _run(N)
        check_lowered_ir("""
        // CHECK-NOT: stage = "distributed"
        // CHECK: linalg.matmul {mlir_edsl.blocked = {{{.*}}stage = "tiled"
        // CHECK-SAME: tensor<4x256xf32>, tensor<256x16xf32>
        // CHECK-NOT: stage = "distributed"
        """, after=TILE_AND_PACK)

    def test_single_iteration_forall_still_reaches_the_kernel(
            self, check_lowered_ir):
        """Canonicalize removes a one-tile forall; the later stages still run.

        Tile-and-pack finds the tile by its attribute, not inside the forall.
        """
        _run(N_SINGLE_TILE)
        check_lowered_ir("""
        // CHECK-NOT: scf.forall
        // CHECK: linalg.matmul {mlir_edsl.blocked = {{{.*}}stage = "kernel"
        // CHECK-SAME: tensor<4x1xf32>, tensor<1x16xf32>
        """, after=KERNEL)


class TestBlockedMatmulFallback:
    """Shapes chooseStrategy rejects must not be touched by the passes."""

    def test_rejected_shape_is_left_for_the_old_path(self, check_lowered_ir):
        """No blocked marker and no 2-D forall for a non-divisible shape."""
        _run(N_REJECTED)
        check_lowered_ir("""
        // CHECK-NOT: mlir_edsl.blocked
        """, after=DISTRIBUTE)

    def test_chained_producer_is_left_and_consumer_blocked(
            self, check_lowered_ir):
        """Blocking one matmul leaves the other candidates intact.

        The inner matmul feeds a linalg op, so the guard rejects it; the outer
        one is blocked after it, from the same up-front candidate list.
        """
        @ml_function
        def chain_fn(A: Tensor[f32, N, N], B: Tensor[f32, N, N],
                     C: Tensor[f32, N, N]) -> Tensor[f32, N, N]:
            return matmul(matmul(A, B), C)

        ones = np.ones((N, N), dtype=np.float32)
        chain_fn(ones, ones, ones)
        check_lowered_ir("""
        // CHECK: linalg.matmul ins({{.*}} : tensor<256x256xf32>, tensor<256x256xf32>)
        // CHECK: scf.forall
        // CHECK: linalg.matmul {mlir_edsl.blocked =
        """, after=DISTRIBUTE)

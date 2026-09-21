"""IR structure tests for LinalgMatmulBlockedPass (INTEGRATION.md Steps 1-2).

A bare f32 matmul whose shape chooseStrategy accepts is tiled into a BLIS loop
nest over a 4x16 register tile, with B packed into contiguous NR-wide panels.
Shapes the guard rejects must keep the old 64x64 / 8x8x8 structure instead.
"""

import numpy as np
from mlir_edsl import ml_function, Tensor, f32, matmul

# 256 is accepted by chooseStrategy: f32, static, square, power of two, no
# linalg consumer. Large enough for every loop level to survive.
N = 256

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
    """The loop nest and packed panel the blocked pass produces."""

    def test_produces_two_dimensional_forall(self, check_lowered_ir):
        """ic and jc are fused into one forall over MC x NC tiles of C."""
        _run(N)
        check_lowered_ir("""
        // CHECK: scf.forall ({{.*}}, {{.*}}) = (0, 0) to (256, 256) step (128, 256)
        """, after="linalg-matmul-blocked")

    def test_register_tile_is_marked_and_4x16x1(self, check_lowered_ir):
        """The innermost tile is MR x NR x 1 and carries the blocked marker."""
        _run(N)
        check_lowered_ir("""
        // CHECK: linalg.matmul {mlir_edsl.blocked}
        // CHECK-SAME: tensor<4x1xf32>, tensor<1x16xf32>
        // CHECK-SAME: tensor<4x16xf32>
        """, after="linalg-matmul-blocked")

    def test_packs_b_into_contiguous_panel(self, check_lowered_ir):
        """B~ is (NC/NR) x KC x NR, built by a vectorized copy loop."""
        _run(N)
        check_lowered_ir("""
        // CHECK: vector.transfer_read
        // CHECK: vector.transfer_write {{.*}} vector<8x16xf32>
        // CHECK-SAME: tensor<256x16xf32>
        """, after="linalg-matmul-blocked")

    def test_microkernel_is_outerproduct(self, check_lowered_ir):
        """The reused pipeline passes turn the register tile into FMAs."""
        _run(N)
        check_lowered_ir("""
        // CHECK: vector.outerproduct
        // CHECK-NOT: linalg.matmul
        """, after="vector-contract-to-outerproduct")


class TestBlockedMatmulFallback:
    """Shapes chooseStrategy rejects must not be touched by the pass."""

    def test_rejected_shape_is_left_for_the_old_path(self, check_lowered_ir):
        """No blocked marker and no 2-D forall for a non-divisible shape."""
        _run(N_REJECTED)
        check_lowered_ir("""
        // CHECK-NOT: mlir_edsl.blocked
        """, after="linalg-matmul-blocked")

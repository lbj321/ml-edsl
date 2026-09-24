"""IR structure tests for the blocked matmul passes (INTEGRATION.md Steps 1-2).

A bare f32 matmul whose shape chooseStrategy accepts is distributed over an
ic x jc forall, tiled into a BLIS loop nest over a 4x16 register tile with A
and B packed into contiguous panels, and k-tiled into the register kernel —
one pass per stage. Extents no cache block divides are padded up to one.
Non-f32 matmuls, which the guard rejects, must keep the old 64x64 / 8x8x8
structure instead.
"""

import numpy as np
from mlir_edsl import ml_function, Tensor, f32, i32, matmul, relu

DISTRIBUTE = "linalg-matmul-blocked-distribute"
TILE = "linalg-matmul-blocked-tile"
PACK = "linalg-matmul-blocked-pack"
KERNEL = "linalg-matmul-blocked-kernel"

# 256 is accepted by chooseStrategy: f32, static, and every block divides.
# Large enough for the forall, jr and ir to survive; pc has one iteration
# (K == KC).
N = 256

# 128 gives MC = NC = 128, so the forall has a single iteration and
# canonicalize removes it between the distribute and tile passes.
N_SINGLE_TILE = 128

# (M, K, N) with M = MR, so ir folds and jr is the only register loop left
# to hoist out of. K = 2 * KC keeps A a real slice.
SHAPE_ONE_REGISTER_LOOP = (4, 512, 256)

# M = MR and N = NR: both register loops fold, so nothing is packed.
SHAPE_NO_REGISTER_LOOP = (4, 64, 16)

# N = NC = NR and K = KC: B's slice is the whole of B, which canonicalize
# folds away, so only A is packed.
N_WHOLE_B = 16

# (M, K, N) that pads every extent: M 98 -> 100 (MR = 4), K 100 -> 104 (8),
# N 100 -> 112 (NR = 16), each a single block.
SHAPE_PADDED = (98, 100, 100)

# Only N needs padding (100 -> 112); M and K divide their blocks.
SHAPE_PADDED_N = (128, 128, 100)


def _run(n):
    """Compile and call an NxN bare matmul, returning nothing."""
    _run_mkn(n, n, n)


def _run_mkn(m, k, n):
    """Compile and call an (MxK) @ (KxN) bare matmul, returning nothing."""
    @ml_function
    def mm_fn(A: Tensor[f32, m, k], B: Tensor[f32, k, n]) -> Tensor[f32, m, n]:
        return matmul(A, B)

    mm_fn(np.ones((m, k), dtype=np.float32), np.ones((k, n), dtype=np.float32))


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
        """, after=PACK)

    def test_packs_a_into_k_major_panel(self, check_lowered_ir):
        """A~ is (MC/MR) x KC x MR, built by a tiled [1,0] transpose."""
        _run(N)
        check_lowered_ir("""
        // CHECK: linalg.transpose ins({{.*}} : tensor<4x8xf32>)
        // CHECK-SAME: outs({{.*}} : tensor<8x4xf32>)
        // CHECK-SAME: permutation = [1, 0]
        """, after=PACK)

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

    def test_tile_marks_register_loops_and_does_not_pack(
            self, check_lowered_ir):
        """jr and ir carry the hoist marker; packing is left to the pack pass."""
        _run(N)
        check_lowered_ir("""
        // CHECK-NOT: tensor.pad
        // CHECK: linalg.matmul {mlir_edsl.blocked = {{{.*}}stage = "tiled"
        // CHECK-SAME: tensor<4x256xf32>, tensor<256x16xf32>
        // CHECK: } {mlir_edsl.blocked_hoist}
        // CHECK: } {mlir_edsl.blocked_hoist}
        // CHECK-NOT: mlir_edsl.blocked_hoist
        """, after=TILE)

    def test_pack_advances_every_tiled_tile(self, check_lowered_ir):
        """No tile is left at the tiled stage after pack."""
        _run(N)
        check_lowered_ir("""
        // CHECK-NOT: stage = "tiled"
        // CHECK: linalg.matmul {mlir_edsl.blocked = {{{.*}}stage = "packed"
        // CHECK-SAME: tensor<4x256xf32>, tensor<256x16xf32>
        // CHECK-NOT: stage = "tiled"
        """, after=PACK)

    def test_pack_removes_hoist_markers(self, check_lowered_ir):
        """The markers are a tile-to-pack hand-over and go no further."""
        _run(N)
        check_lowered_ir("""
        // CHECK-NOT: mlir_edsl.blocked_hoist
        """, after=PACK)

    def test_single_iteration_forall_still_reaches_the_kernel(
            self, check_lowered_ir):
        """Canonicalize removes a one-tile forall; the later stages still run.

        The tile pass finds the tile by its attribute, not inside the forall.
        """
        _run(N_SINGLE_TILE)
        check_lowered_ir("""
        // CHECK-NOT: scf.forall
        // CHECK: linalg.matmul {mlir_edsl.blocked = {{{.*}}stage = "kernel"
        // CHECK-SAME: tensor<4x1xf32>, tensor<1x16xf32>
        """, after=KERNEL)


class TestBlockedMatmulPackingDecisions:
    """The pack pass hoists out of the register loops canonicalize left."""

    def test_one_register_loop_still_packs_both_operands(
            self, check_lowered_ir):
        """With only jr left, A and B are still packed, hoisted out of jr."""
        _run_mkn(*SHAPE_ONE_REGISTER_LOOP)
        check_lowered_ir("""
        // CHECK-DAG: vector.transfer_write {{.*}} vector<8x16xf32>
        // CHECK-DAG: linalg.transpose ins({{.*}} : tensor<4x8xf32>)
        // CHECK: linalg.matmul {mlir_edsl.blocked = {{{.*}}stage = "packed"
        """, after=PACK)

    def test_one_register_loop_fuses_the_untranspose(self, check_lowered_ir):
        """The kernel pass finds the un-transpose packing A left behind."""
        _run_mkn(*SHAPE_ONE_REGISTER_LOOP)
        check_lowered_ir("""
        // CHECK: linalg.transpose ins({{.*}} : tensor<1x4xf32>)
        // CHECK-SAME: outs({{.*}} : tensor<4x1xf32>)
        """, after=KERNEL)

    def test_no_register_loop_skips_packing(self, check_lowered_ir):
        """Each panel would feed a single microtile, so nothing is packed."""
        _run_mkn(*SHAPE_NO_REGISTER_LOOP)
        check_lowered_ir("""
        // CHECK-NOT: tensor.pad
        // CHECK-NOT: linalg.transpose
        // CHECK: linalg.matmul {mlir_edsl.blocked = {{{.*}}stage = "packed"
        // CHECK-NOT: tensor.pad
        // CHECK-NOT: linalg.transpose
        """, after=PACK)

    def test_no_register_loop_reaches_the_kernel(self, check_lowered_ir):
        """An unpacked tile is still k-tiled, with no un-transpose to fuse."""
        _run_mkn(*SHAPE_NO_REGISTER_LOOP)
        check_lowered_ir("""
        // CHECK-NOT: linalg.transpose
        // CHECK: linalg.matmul {mlir_edsl.blocked = {{{.*}}stage = "kernel"
        // CHECK-SAME: tensor<4x1xf32>, tensor<1x16xf32>
        """, after=KERNEL)

    def test_whole_tensor_operand_is_not_packed(self, check_lowered_ir):
        """B's full-size slice folds away, so only A is packed."""
        _run(N_WHOLE_B)
        check_lowered_ir("""
        // CHECK-NOT: vector.transfer_write
        // CHECK: linalg.transpose ins({{.*}} : tensor<4x8xf32>)
        // CHECK-NOT: vector.transfer_write
        // CHECK: linalg.matmul {mlir_edsl.blocked = {{{.*}}stage = "packed"
        """, after=PACK)


class TestBlockedMatmulConsumers:
    """A matmul's consumers do not keep it off the blocked path."""

    def test_both_matmuls_in_a_chain_are_blocked(self, check_lowered_ir):
        """Blocking the inner matmul leaves the outer candidate intact, and
        both are blocked from the same up-front candidate list."""
        @ml_function
        def chain_fn(A: Tensor[f32, N, N], B: Tensor[f32, N, N],
                     C: Tensor[f32, N, N]) -> Tensor[f32, N, N]:
            return matmul(matmul(A, B), C)

        ones = np.ones((N, N), dtype=np.float32)
        chain_fn(ones, ones, ones)
        check_lowered_ir("""
        // CHECK-NOT: linalg.matmul ins(
        // CHECK-COUNT-2: linalg.matmul {mlir_edsl.blocked =
        // CHECK-NOT: linalg.matmul ins(
        """, after=DISTRIBUTE)

    def test_dense_layer_matmul_is_blocked_and_epilogue_unfused(
            self, check_lowered_ir):
        """relu(matmul + b) blocks the matmul down to the register kernel;
        bias and relu run after it as their own ops."""
        @ml_function
        def dense(A: Tensor[f32, N_SINGLE_TILE, N_SINGLE_TILE],
                  B: Tensor[f32, N_SINGLE_TILE, N_SINGLE_TILE],
                  b: Tensor[f32, N_SINGLE_TILE]
                  ) -> Tensor[f32, N_SINGLE_TILE, N_SINGLE_TILE]:
            return relu(matmul(A, B) + b)

        n = N_SINGLE_TILE
        dense(np.ones((n, n), dtype=np.float32),
              np.ones((n, n), dtype=np.float32),
              np.ones(n, dtype=np.float32))
        check_lowered_ir("""
        // CHECK: linalg.matmul {mlir_edsl.blocked = {{{.*}}stage = "kernel"
        // CHECK: library_call = "bias_add"
        // CHECK: library_call = "relu"
        """, after=KERNEL)


class TestBlockedMatmulPadding:
    """Extents no cache block divides are padded up to one, not rejected."""

    def test_every_extent_padded_to_its_block(self, check_lowered_ir):
        """A and B are copied into padded buffers, the forall covers the
        padded C, and the result is sliced back out of it."""
        _run_mkn(*SHAPE_PADDED)
        check_lowered_ir("""
        // CHECK: tensor.empty() : tensor<100x104xf32>
        // CHECK: tensor.empty() : tensor<104x112xf32>
        // CHECK: scf.forall ({{.*}}, {{.*}}) = (0, 0) to (100, 112)
        // CHECK: linalg.matmul {mlir_edsl.blocked = {kc = 104 : i64, mc = 100 : i64
        // CHECK: tensor.extract_slice {{.*}}[0, 0] [98, 100] [1, 1] : tensor<100x112xf32> to tensor<98x100xf32>
        """, after=DISTRIBUTE)

    def test_only_the_failing_extent_is_padded(self, check_lowered_ir):
        """With only N padded, A is used as it is and only B and C get
        padded buffers."""
        _run_mkn(*SHAPE_PADDED_N)
        check_lowered_ir("""
        // CHECK-NOT: tensor<128x128xf32> to tensor
        // CHECK: tensor.empty() : tensor<128x112xf32>
        // CHECK: scf.forall ({{.*}}, {{.*}}) = (0, 0) to (128, 112)
        """, after=DISTRIBUTE)

    def test_divisible_shape_is_not_padded(self, check_lowered_ir):
        """Blocks that divide every extent leave the operands untouched: the
        only tensor.empty is the one the EDSL's C fill writes into."""
        _run(N)
        check_lowered_ir("""
        // CHECK: tensor.empty() : tensor<256x256xf32>
        // CHECK-NOT: tensor.empty
        // CHECK: scf.forall ({{.*}}, {{.*}}) = (0, 0) to (256, 256)
        // CHECK-NOT: tensor.empty
        """, after=DISTRIBUTE)


class TestBlockedMatmulFallback:
    """Matmuls chooseStrategy rejects must not be touched by the passes."""

    def test_non_f32_matmul_is_left_for_the_old_path(self, check_lowered_ir):
        """An i32 matmul gets no blocked marker."""
        @ml_function
        def mm_fn(A: Tensor[i32, N, N], B: Tensor[i32, N, N]) -> Tensor[i32, N, N]:
            return matmul(A, B)

        ones = np.ones((N, N), dtype=np.int32)
        mm_fn(ones, ones)
        check_lowered_ir("""
        // CHECK-NOT: mlir_edsl.blocked
        """, after=DISTRIBUTE)

    def test_unblocked_matmul_is_left_for_scalar_loops(self, check_lowered_ir):
        """Neither the matmul nor its fill is vectorized whole: both reach
        convert-linalg-to-loops as they are."""
        @ml_function
        def mm_fn(A: Tensor[i32, N, N], B: Tensor[i32, N, N]) -> Tensor[i32, N, N]:
            return matmul(A, B)

        ones = np.ones((N, N), dtype=np.int32)
        mm_fn(ones, ones)
        check_lowered_ir("""
        // CHECK-NOT: vector<256x256xi32>
        // CHECK: linalg.fill
        // CHECK: linalg.matmul
        // CHECK-NOT: vector.contract
        """, after="linalg-vectorize")

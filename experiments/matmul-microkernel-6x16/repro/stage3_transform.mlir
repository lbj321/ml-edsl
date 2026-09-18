// Stage 3, approach 1 (global pack first): pack the whole matmul at the
// microkernel's tile size, then tile the resulting packed generic through
// all five loop levels (jc -> pc -> ic -> jr -> ir), then apply Stage 1's
// exact K-tile/vectorize/lower/hoist recipe to the innermost tile.
//
// Blocking for this script's test problem (M=336, N=64, K=512):
//   MC=168 (28 MR-blocks per ic chunk), NC=32 (2 NR-blocks per jc chunk),
//   KC=256 (2 pc chunks total).
//
// Three bugs found in review before this version, all fixed below:
//
//   1. No FMA formed. transform.structured.vectorize (unlike Stage 1's
//      vectorize_children_and_apply_patterns, which runs reduction-to-
//      contract patterns immediately) leaves the reduction as plain
//      vector.multi_reduction / arith.mulf+addf. Left alone, canonicalize
//      folds the size-1 k reduction straight to mul/add before any
//      vector.contract can form, so lower_contraction/lower_outerproduct
//      have nothing to act on -- confirmed empirically (0 vfmadd231ps,
//      needed `llc -fp-contract=fast` as a stopgap, which also left the
//      A-broadcast illegal-<6xfloat>-load problem from Stage 1 unfixed).
//      Fix: run apply_patterns.vector.reduction_to_contract in its own
//      phase, before cast_away_vector_leading_one_dim/canonicalization.
//
//   2. B was never actually packed into a K-major panel. packed_sizes
//      packs N in place: B's original shape is (K, N), N is not B's
//      leading dim, so packing appends the new tile-size dim (ni) at the
//      end but leaves K in front -- result is (K, no, ni), a reshape of
//      the ORIGINAL array with zero data movement (lower_pack emits only
//      an expand_shape, no transpose -- the tell). That gives the k-loop a
//      64-float (256B) stride instead of walking a contiguous panel, i.e.
//      the same L1 set-conflict pattern Stage 2 hit. A worked by
//      coincidence only because M already was A's leading dim. Fix:
//      transform.structured.pack_transpose on B's pack with outer_perm =
//      [1, 0] to swap it to (no, k, ni), matching A's per-tile-contiguous
//      layout.
//
//   3. C gets packed too (packed_sizes packs every operand along the
//      tiled dims) and unpacked at the end via transpose + collapse_shape
//      + linalg.copy -- two extra full passes over C that aren't part of
//      real BLIS (which updates C in place with stride ldc). Accepted as
//      known overhead for this approach-1 layout/correctness checkpoint;
//      benchmark with and without the unpack region timed to quantify it
//      separately. (Packing only A/B via transform.structured.pad +
//      hoist_pad, leaving C untouched, is the fix for approach 2.)
//
// Plus one more found independently by actually running this to assembly
// (not in the review above): lower_pack/lower_unpack's linalg.transpose
// ops are never vectorized, so -convert-linalg-to-loops lowers the actual
// panel-packing data movement as scalar vmovss-one-float-at-a-time loops
// -- confirmed as the dominant cost (correctness held at three problem
// sizes, but GFLOPS was 30-44 regardless of size, well under the ~80-90%-
// of-Stage-2 estimate). Fixed by vectorizing the transpose ops too, not
// just the microkernel tile.
//
// Two more bugs found in the next review round:
//
//   4. Fix 1 (reduction_to_contract alone) was necessary but not
//      sufficient: reduction_to_contract couldn't match because the
//      transfer_reads were still indexed over all five tiled dims, not
//      folded down to A:(mo,mi,k)/B:(no,ni,k) via their permutation maps.
//      Confirmed by print 2/3 showing 96 scalar mulf/addf pairs (6x16
//      unrolled) instead of a contract. Fix: run
//      apply_patterns.vector.transfer_permutation_patterns in the same
//      phase, before reduction_to_contract.
//
//   5. transform.structured.vectorize on the untiled pack/unpack
//      transposes (bug from note above, actually applied wrong) produces
//      whole-array vectors -- vector<56x512x6xf32> for A, vector<4x512x16xf32>
//      for B, a 4-D vector for all of C -- the same whole-array-vector
//      failure mode, just moved from the kernel to the packing step. Fix:
//      tile the transposes (using lower_pack/lower_unpack's own returned
//      handles, not a re-match) before vectorizing them.
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %mm = transform.structured.match ops{["linalg.matmul"]} in %root
      : (!transform.any_op) -> !transform.any_op

    %packed = transform.structured.pack %mm packed_sizes = [6, 16, 0]
      : (!transform.any_op) -> !transform.op<"linalg.generic">

    // Fix 2: B is operand 1 (ins: A=0, B=1; outs: C=2). Swap its pack to
    // (no, k, ni) so each NR-panel is contiguous, matching A's layout.
    %b_pack = transform.get_producer_of_operand %packed[1]
      : (!transform.op<"linalg.generic">) -> !transform.op<"linalg.pack">
    %packed_t, %b_pack_t, %b_unpack_t = transform.structured.pack_transpose %b_pack
        with_compute_op(%packed) outer_perm = [1, 0]
      : (!transform.op<"linalg.pack">, !transform.op<"linalg.generic">)
        -> (!transform.op<"linalg.generic">, !transform.op<"linalg.pack">, !transform.any_op)

    // Fix 4 (found in second review, replacing the earlier reduction_to_
    // contract-only fix which was necessary but not sufficient): the pack
    // leaves iterator order (mo, no, k, mi, ni) -- reduction (k) in the
    // MIDDLE. lower_contraction's "outerproduct" strategy only matches a
    // 3-D contract with iterators [parallel, parallel, reduction] (k
    // last), the same order Stage 1's named matmul_transpose_a already
    // had. With k in the middle, lower_contraction falls back to
    // unrolling every parallel dim down to scalars (confirmed empirically:
    // 96 = 6x16 individual vector<1xf32> mulf/addf sequences instead of
    // vector.fma). Interchange so k is last before tiling: new order
    // (mo, no, mi, ni, k) = old indices [0, 1, 3, 4, 2].
    %inter = transform.structured.interchange %packed_t iterator_interchange = [0, 1, 3, 4, 2]
      : (!transform.op<"linalg.generic">) -> !transform.op<"linalg.generic">

    // Tile sizes now target (mo, no, mi, ni, k) -- k moved to the last slot.
    // jc: tile no (NC/NR = 32/16 = 2 NR-blocks per jc chunk), outermost.
    %t_jc, %jc = transform.structured.tile_using_for %inter tile_sizes [0, 2, 0, 0, 0]
      : (!transform.op<"linalg.generic">) -> (!transform.any_op, !transform.any_op)
    // pc: tile k (real K extent, KC=256), nested in jc.
    %t_pc, %pc = transform.structured.tile_using_for %t_jc tile_sizes [0, 0, 0, 0, 256]
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    // ic: tile mo (MC/MR = 168/6 = 28 MR-blocks per ic chunk), nested in pc.
    %t_ic, %ic = transform.structured.tile_using_for %t_pc tile_sizes [28, 0, 0, 0, 0]
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    // jr: tile no (now ranges over the jc chunk's 2 NR-blocks) by 1, nested in ic.
    %t_jr, %jr = transform.structured.tile_using_for %t_ic tile_sizes [0, 1, 0, 0, 0]
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    // ir: tile mo (now ranges over the ic chunk's 28 MR-blocks) by 1, nested in jr.
    %t_ir, %ir = transform.structured.tile_using_for %t_jr tile_sizes [1, 0, 0, 0, 0]
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

    // Stage 1's recipe on the remaining (mi=6, ni=16, k=KC) tile: tile k
    // (now last, d4) by 1, vectorize, form the contract, lower contraction
    // -> outerproduct, hoist.
    %t_k, %kloop = transform.structured.tile_using_for %t_ir tile_sizes [0, 0, 0, 0, 1]
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

    transform.structured.vectorize %t_k : !transform.any_op

    %f = transform.structured.match ops{["func.func"]} in %root
      : (!transform.any_op) -> !transform.any_op
    transform.print %f {name = "print 1: after vectorize, before reduction_to_contract (expect vector.multi_reduction)"} : !transform.any_op

    // Fix 1: form vector.contract from vector.multi_reduction in its own
    // phase, before anything else gets a chance to canonicalize the size-1
    // k reduction straight down to plain mul/add. transfer_permutation_patterns
    // runs first in the same phase so the broadcast dims folded into the
    // transfer_read permutation maps (A: (mo,mi,k), B: (no,ni,k)) get pulled
    // out before reduction_to_contract tries to match -- without it,
    // reduction_to_contract sees transfer_reads still indexed over all five
    // tiled dims and can't form a clean 3-D contract.
    transform.apply_patterns to %f {
      transform.apply_patterns.vector.transfer_permutation_patterns
      transform.apply_patterns.vector.reduction_to_contract
    } : !transform.any_op
    transform.print %f {name = "print 2: after reduction_to_contract (expect vector.contract with iterators ending in reduction)"} : !transform.any_op

    transform.apply_patterns to %f {
      // Tiling mo/no down to 1 each (the ic/jc -> ir/jr steps) leaves the
      // tile's operands with leading size-1 dims (e.g. tensor<1x1x6x16xf32>
      // instead of tensor<6x16xf32>); cast those away so the contraction/
      // outerproduct lowering sees the same clean 2D shapes Stage 1 did.
      transform.apply_patterns.vector.cast_away_vector_leading_one_dim
      transform.apply_patterns.vector.lower_contraction lowering_strategy = "outerproduct"
      transform.apply_patterns.vector.lower_outerproduct
      transform.apply_patterns.canonicalization
    } : !transform.any_op
    transform.print %f {name = "print 3: after lowering (expect six vector.fma on vector<16xf32>)"} : !transform.any_op

    %forops = transform.structured.match ops{["scf.for"]} in %f
      : (!transform.any_op) -> !transform.any_op
    transform.loop.hoist_loop_invariant_subsets %forops : !transform.any_op

    // linalg.pack/unpack have no bufferization interface implementation on
    // the pinned commit (one-shot-bufferize fails with "op was not
    // bufferized" otherwise) -- decompose them into pad/expand_shape/
    // transpose (pack) and empty/transpose/collapse_shape/extract_slice
    // (unpack), which bufferize the ordinary way.
    %packs = transform.structured.match ops{["linalg.pack"]} in %root
      : (!transform.any_op) -> !transform.op<"linalg.pack">
    %p_pad, %p_exp, %p_tr = transform.structured.lower_pack %packs
      : (!transform.op<"linalg.pack">)
        -> (!transform.op<"tensor.pad">, !transform.op<"tensor.expand_shape">, !transform.op<"linalg.transpose">)
    %unpacks = transform.structured.match ops{["linalg.unpack"]} in %root
      : (!transform.any_op) -> !transform.op<"linalg.unpack">
    %u_empty, %u_tr, %u_col, %u_slice = transform.structured.lower_unpack %unpacks
      : (!transform.op<"linalg.unpack">)
        -> (!transform.op<"tensor.empty">, !transform.op<"linalg.transpose">, !transform.op<"tensor.collapse_shape">, !transform.op<"tensor.extract_slice">)

    // Fix 5: transform.structured.vectorize on the untiled pack/unpack
    // transposes produces whole-array vectors (vector<56x512x6xf32> for A,
    // vector<4x512x16xf32> for B, a 4-D vector for all of C) -- the same
    // failure mode as a whole-array-read, just relocated to the packing
    // step instead of the kernel. Tile first, then vectorize only the tile.
    // A and B's lower_pack transposes are both 3-D (56x512x6, 4x512x16) --
    // one tile shape covers both: 1 x 8 x (full inner dim).
    %p_tiled, %p_l0, %p_l1 = transform.structured.tile_using_for %p_tr tile_sizes [1, 8, 0]
      : (!transform.op<"linalg.transpose">) -> (!transform.any_op, !transform.any_op, !transform.any_op)
    // C's lower_unpack transpose is 4-D (56x6x4x16); the inner 16 stays
    // contiguous (plain row copy, not a real transpose), so tile the other
    // three dims down to 1 each.
    %u_tiled, %u_l0, %u_l1, %u_l2 = transform.structured.tile_using_for %u_tr tile_sizes [1, 1, 1, 0]
      : (!transform.op<"linalg.transpose">) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)

    transform.structured.vectorize %p_tiled : !transform.any_op
    transform.structured.vectorize %u_tiled : !transform.any_op

    // Fix 6: found by benchmarking (36 GFLOPS, way under the 80-90%-of-
    // Stage-2 estimate). Tiling to [1, 8, 0] leaves the packing tile's
    // vector.transfer_write with a leading size-1 dim (e.g.
    // vector<1x8x16xf32>, not vector<8x16xf32>) -- unlike the kernel tile,
    // which had cast_away_vector_leading_one_dim run on it explicitly.
    // convert-vector-to-scf can't lower a >2-D transfer_write directly, so
    // it falls back to materializing the vector into a stack alloca and
    // memref.copy-ing it out, which finalize-memref-to-llvm turns into an
    // actual `callq memcpy@PLT` inside the hot packing loop -- confirmed
    // as the dominant cost by reading the assembly (B's pack loop, unrolled
    // 4x, calls memcpy once per iteration). Cast away the leading unit dim
    // in the same phase as the other transpose-lowering patterns so
    // convert-vector-to-llvm sees plain 2-D vectors instead.
    %f2 = transform.structured.match ops{["func.func"]} in %root
      : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %f2 {
      transform.apply_patterns.vector.transfer_permutation_patterns
      transform.apply_patterns.vector.cast_away_vector_leading_one_dim
      transform.apply_patterns.vector.lower_transpose
      transform.apply_patterns.canonicalization
    } : !transform.any_op
    transform.print %f2 {name = "print 4: after transpose tiling/vectorization (expect no vector type >16 elements/dim outside the kernel)"} : !transform.any_op

    transform.yield
  }
}

// Stage 6: multithreaded matmul. stage3b_transform.mlir (Stage 3 approach 2,
// per-block packing via pad + hoist_pad) with three changes:
//
//   1. jc (the outermost loop) is tile_using_forall instead of tile_using_for,
//      with NC = N / 8 -> one jc block per thread. Each thread packs its own
//      B~ and A~ (allocated inside the forall body after bufferization) and
//      writes a disjoint column block of C, so there are no races.
//      Why jc and not ic (BLIS's usual choice): hoist_pad only hoists across
//      scf.for, and B~ is hoisted *above* ic, so ic can't be an scf.forall;
//      and there is no loop.for_to_forall at the pin to convert it afterwards.
//      Cost: every thread packs all of A once per pc (8x redundant A packing).
//   2. 4x16 microkernel (MR=4, MC=128) instead of 6x16 -- divides every
//      power-of-2 size (PLAN.md Stage 5).
//   3. NC=128, KC=256 fixed for the 1024x1024x1024 input. For other
//      power-of-2 sizes N, set NC = N / 8:
//        sed 's/tile_sizes \[0, 128, 0\]/tile_sizes [0, 64, 0]/'   # N=512
//        sed 's/tile_sizes \[0, 128, 0\]/tile_sizes [0, 32, 0]/'   # N=256
//      (MC=128 and KC=256 need M, K to be multiples of them, or M/K
//      themselves when smaller: sed the ic/pc tile sizes too.)
//
// Comments below that say "6" / "6x16" / "6xKC" mean MR / MRxNR / MRxKC --
// they were written for the original 6x16 kernel.
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %mm = transform.structured.match ops{["linalg.matmul"]} in %root
      : (!transform.any_op) -> !transform.any_op

    // Tile sizes are [m, n, k]; one non-zero entry per call fixes loop order
    // jc -> pc -> ic -> jr -> ir.
    // jc is the parallel loop: scf.forall with NC = N / 8 (one column block
    // per thread). Everything below is the unchanged serial Stage 3 recipe,
    // running inside the forall body.
    %t_jc, %jc = transform.structured.tile_using_forall %mm tile_sizes [0, 128, 0]
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    %t_pc, %pc = transform.structured.tile_using_for %t_jc tile_sizes [0, 0, 256]
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    %t_ic, %ic = transform.structured.tile_using_for %t_pc tile_sizes [128, 0, 0]
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    %t_jr, %jr = transform.structured.tile_using_for %t_ic tile_sizes [0, 16, 0]
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    %t_ir, %ir = transform.structured.tile_using_for %t_jr tile_sizes [4, 0, 0]
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

    // Each tile level slices the previous level's slice, so the microtile's
    // A/B operand is defined *inside* the ic/jr loops. hoist_pad's analysis
    // requires the pad's source slice to read something defined above the
    // outermost hoisted loop ("Source not defined outside of loops -> Skip"),
    // so collapse the extract_slice chains down to one slice of the original.
    %f0 = transform.structured.match ops{["func.func"]} in %root
      : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %f0 {
      transform.apply_patterns.tensor.merge_consecutive_insert_extract_slice
      transform.apply_patterns.canonicalization
    } : !transform.any_op

    // nofold_flags [1, 1, 0]: A and B pads must survive as real copies even
    // though the slices are already exactly 6xKC / KCx16; C's pad folds away.
    %padded, %pads, %copy_back = transform.structured.pad %t_ir {
      padding_values = [0.0 : f32, 0.0 : f32, 0.0 : f32],
      padding_dimensions = [0, 1, 2],
      nofold_flags = [1, 1, 0],
      copy_back_op = "none"
    } : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)

    %f = transform.structured.match ops{["func.func"]} in %root
      : (!transform.any_op) -> !transform.any_op

    // B~: hoist above ir, jr, ic. Only jr indexes B's slice, so jr is the one
    // packing dim -> tensor<(NC/16) x KC x 16> built at pc level.
    %pad_b = transform.get_producer_of_operand %padded[1]
      : (!transform.any_op) -> !transform.any_op
    %hoisted_b = transform.structured.hoist_pad %pad_b by 3 loops
      : (!transform.any_op) -> !transform.any_op

    // A~: hoist above ir, jr, transposed to k-major micro-panels ->
    // tensor<(MC/6) x KC x 6> built at ic level.
    %pad_a = transform.get_producer_of_operand %padded[0]
      : (!transform.any_op) -> !transform.any_op
    %hoisted_a = transform.structured.hoist_pad %pad_a by 2 loops, transpose by [1, 0]
      : (!transform.any_op) -> !transform.any_op

    // ---- Packing loops: tile to row-sized chunks before vectorizing ----
    // (Approach 1's Fix 5 lesson: an untiled copy vectorizes to one
    // whole-panel vector.)
    // B~: pad(KCx16 slice of B) is a plain row copy. Tile it k by 8 and
    // vectorize each 8x16 pad tile (transfer_read B -> transfer_write).
    %b_pad_t, %b_pad_loop = transform.structured.tile_using_for %hoisted_b tile_sizes [8, 0]
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.structured.vectorize %b_pad_t vector_sizes [8, 16] : !transform.any_op
    // The vectorized pad is transfer_read(B) -> transfer_write(tensor.empty)
    // -> insert_slice into the B~ tile. Fold the insert_slice into the
    // transfer_write NOW, before any canonicalization: otherwise canonicalize
    // folds the read/write round trip through the fresh empty tensor back to
    // insert_slice(extract_slice(B)), which bufferizes to a strided
    // memref.copy -> a call to the `memrefCopy` runtime helper (undefined
    // symbol at dlopen, and a slow generic element loop even if linked).
    transform.apply_patterns to %f {
      transform.apply_patterns.tensor.fold_tensor_subset_ops_into_vector_transfers
    } : !transform.any_op

    // A~: pad(6xKC slice of A) -> transpose -> (KCx6 slice of A~). The pad is
    // zero-width, i.e. a redundant copy -- the transpose already is the
    // packing copy. (fuse_into_containing_op can't pull the pad into the
    // transpose's tile loop: it asserts on non-destination-style ops like
    // tensor.pad.) decompose_pad turns it into empty+fill+insert_slice, and
    // the full-size insert_slice folds to its source, so the transpose reads
    // A directly. Then tile the transpose along k by 8: 6x8 read -> 8x6 write.
    %a_tr = transform.get_consumers_of_result %hoisted_a[0]
      : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %f {
      transform.apply_patterns.linalg.decompose_pad
      transform.apply_patterns.canonicalization
    } : !transform.any_op
    %a_tr_t, %a_tr_loop = transform.structured.tile_using_for %a_tr tile_sizes [8, 0]
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

    transform.print %f {name = "print 3: after tiling packing ops"} : !transform.any_op

    // ---- Microkernel: Stage 2's recipe ----
    %mm2 = transform.structured.match ops{["linalg.matmul"]} in %root
      : (!transform.any_op) -> !transform.any_op
    // hoist_pad(transpose) leaves an un-transpose (KCx6 -> 6xKC) feeding the
    // matmul every microtile. Fuse it into the k-loop so each k step
    // transposes a 1x6 row of A~ -- i.e. the k-major read Stage 1 does.
    %untr = transform.get_producer_of_operand %mm2[0]
      : (!transform.any_op) -> !transform.any_op
    %t_k, %kloop = transform.structured.tile_using_for %mm2 tile_sizes [0, 0, 1]
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    %untr_t, %kloop2 = transform.structured.fuse_into_containing_op %untr into %kloop
      : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)

    transform.print %f {name = "print 4: after k-tiling + un-transpose fusion"} : !transform.any_op

    // Vectorize only the three compute tiles -- NOT the whole function via
    // vectorize_children_and_apply_patterns. On this commit tensor.insert_slice
    // has a vectorization impl, so vectorize_children also turns every tiling
    // level's result insert_slice into a whole-block vector copy
    // (vector<336x32>, vector<168x16>, ...). (Stage 2 had the same
    // vector<168x16> copy, harmless there; here it would be at every level.)
    transform.structured.vectorize %a_tr_t : !transform.any_op
    transform.structured.vectorize %untr_t : !transform.any_op
    transform.structured.vectorize %t_k : !transform.any_op

    // structured.vectorize (unlike vectorize_children) leaves the k
    // reduction as vector.multi_reduction -- form the contract in its own
    // phase before canonicalization folds the size-1 reduction to mul/add
    // (approach 1's Fixes 1 and 4).
    transform.apply_patterns to %f {
      transform.apply_patterns.vector.transfer_permutation_patterns
      transform.apply_patterns.vector.reduction_to_contract
    } : !transform.any_op
    transform.apply_patterns to %f {
      transform.apply_patterns.tensor.fold_tensor_subset_ops_into_vector_transfers
      transform.apply_patterns.vector.cast_away_vector_leading_one_dim
      transform.apply_patterns.vector.lower_contraction lowering_strategy = "outerproduct"
      transform.apply_patterns.vector.lower_outerproduct
      transform.apply_patterns.vector.lower_transpose
      transform.apply_patterns.canonicalization
    } : !transform.any_op

    %forops = transform.structured.match ops{["scf.for"]} in %f
      : (!transform.any_op) -> !transform.any_op
    transform.loop.hoist_loop_invariant_subsets %forops : !transform.any_op

    transform.print %f {name = "print 5: after vectorize + lower + hoist"} : !transform.any_op

    transform.yield
  }
}

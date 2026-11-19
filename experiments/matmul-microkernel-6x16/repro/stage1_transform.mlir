// Stage 1 transform script: tile K by 1, vectorize, lower vector.contract
// via the outerproduct strategy, lower outerproducts to vector.fma, then
// hoist the C accumulator out of the k-loop.
//
// Run with upstream `mlir-opt` (not mlir-edsl-opt -- the project's custom
// opt tool doesn't register the Transform dialect or its extensions, see
// PLAN.md's "Harness precedent" note; this stage needs only upstream
// transform ops, same as experiments/matmul-bias-relu-tile-fuse/repro).
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %mm = transform.structured.match ops{["linalg.matmul_transpose_a"]} in %root
      : (!transform.any_op) -> !transform.any_op

    %tiled, %kloop = transform.structured.tile_using_for %mm tile_sizes [0, 0, 1]
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

    %f = transform.structured.match ops{["func.func"]} in %root
      : (!transform.any_op) -> !transform.any_op
    %fv = transform.structured.vectorize_children_and_apply_patterns %f
      : (!transform.any_op) -> !transform.any_op

    transform.apply_patterns to %fv {
      transform.apply_patterns.vector.lower_contraction lowering_strategy = "outerproduct"
      transform.apply_patterns.vector.lower_outerproduct
      transform.apply_patterns.canonicalization
    } : !transform.any_op

    // Re-match the (now vectorized) k-loop and hoist the loop-invariant C
    // accumulator subset (transfer_read/write pair) out of it. This is what
    // decides whether the kernel spills: left inside, C gets loaded/stored
    // on every k-iteration instead of just once before/after the loop.
    %forops = transform.structured.match ops{["scf.for"]} in %fv
      : (!transform.any_op) -> !transform.any_op
    transform.loop.hoist_loop_invariant_subsets %forops : !transform.any_op

    transform.yield
  }
}

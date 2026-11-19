// Stage 2 transform script: wrap the Stage 1 microkernel recipe in two
// explicit register-block loops, tiled one dimension at a time (as PLAN.md's
// Stage 3 note recommends) so the loop order is controlled directly instead
// of relying on tile_using_for's default outer-to-inner dimension order:
//
//   1. tile N by NR=16   -> jr loop (outer)
//   2. tile M by MR=6    -> ir loop (inner, nested in jr)
//   3. tile K by 1, vectorize, lower contraction/outerproduct, hoist
//      -- exactly Stage 1's recipe, now applied to the 6x16x256 tile that
//      steps 1-2 leave behind.
//
// Run with upstream `mlir-opt` (same reason as Stage 1: mlir-edsl-opt
// doesn't register the Transform dialect).
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %mm = transform.structured.match ops{["linalg.matmul_transpose_a"]} in %root
      : (!transform.any_op) -> !transform.any_op

    // jr: tile N (register block NR=16), outer loop.
    %tiled_n, %jr = transform.structured.tile_using_for %mm tile_sizes [0, 16, 0]
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

    // ir: tile M (register block MR=6), inner loop, nested in jr.
    %tiled_mn, %ir = transform.structured.tile_using_for %tiled_n tile_sizes [6, 0, 0]
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

    // Stage 1's recipe, applied to the resulting 6x16x256 tile: tile K by 1,
    // vectorize, lower contraction -> outerproduct, hoist the accumulator.
    %tiled_mnk, %kloop = transform.structured.tile_using_for %tiled_mn tile_sizes [0, 0, 1]
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

    %forops = transform.structured.match ops{["scf.for"]} in %fv
      : (!transform.any_op) -> !transform.any_op
    transform.loop.hoist_loop_invariant_subsets %forops : !transform.any_op

    transform.yield
  }
}

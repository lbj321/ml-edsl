// Transform-dialect library: tile the block-packed matmul generic's outer
// M/N block-grid dims (parallel iterators) to 1 via scf.forall, leaving K
// block-grid and the 32x32x32 inner tile untouched. Runs as its own
// --transform-interpreter pass over out/packed.mlir.
//
// -linalg-block-pack-matmul with block-factors=32,32,32 produces a rank-6
// generic (Mo,No,Ko,Mi,Ni,Ki) - tile_sizes below target dims in that order.
// scf.forall (not tile_using_for) so the OpenMP conversion path (forall ->
// parallel -> omp.parallel, later in run.sh) has something to act on.
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%module: !transform.any_op {transform.readonly}) {
    %generic = transform.structured.match ops{["linalg.generic"]} in %module
        : (!transform.any_op) -> !transform.any_op
    %tiled_generic, %forall =
        transform.structured.tile_using_forall %generic tile_sizes [1, 1, 0, 0, 0, 0]
        : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

    %func = transform.structured.match ops{["func.func"]} attributes{sym_name = "matmul_fn_0"} in %module
        : (!transform.any_op) -> !transform.any_op

    // Tiling-by-one leaves the tiled generic at rank 6 with two unit-extent
    // dims (1x1x32x32x32x32) - fold those away so the next stage sees a
    // clean rank-4 generic.
    transform.apply_patterns to %func {
      transform.apply_patterns.canonicalization
      transform.apply_patterns.linalg.tiling_canonicalization
    } : !transform.any_op
    transform.apply_cse to %func : !transform.any_op
    transform.apply_patterns to %func {
      transform.apply_patterns.linalg.fold_unit_extent_dims_via_reshapes
    } : !transform.any_op

    transform.yield
  }
}

// Transform-dialect library: serial K block-grid tiling (32 -> 1), inside
// the M/N forall body left by tile_mn_forall.mlir. Reduction dim, so
// tile_using_for (iter_args accumulation), not forall - matches
// LinalgMatmulKTilingPass running nested inside the outer parallel tile in
// the real compiler.
//
// tile_mn_forall.mlir's fold_unit_extent_dims_via_reshapes step already
// dropped the M/N block-grid dims, so the generic here is rank 4
// (Ko, Mi, Ni, Ki) - tile_sizes below target dim 0 (Ko).
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%module: !transform.any_op {transform.readonly}) {
    %generic = transform.structured.match ops{["linalg.generic"]} in %module
        : (!transform.any_op) -> !transform.any_op
    %tiled_generic, %kb_loop =
        transform.structured.tile_using_for %generic tile_sizes [1, 0, 0, 0]
        : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

    %func = transform.structured.match ops{["func.func"]} attributes{sym_name = "matmul_fn_0"} in %module
        : (!transform.any_op) -> !transform.any_op

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

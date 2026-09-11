// Transform-dialect library: second-level tile 32x32x32 -> 8x8x8, matching
// the real compiler's vectorization granularity (LinalgMatmulTilingPass(8,8,8)
// / LinalgMatmulToContractPass). Vectorizing directly at 32x32x32 works but
// blows up SSA count (~10x, per matmul-per-tile-packing's measurement).
//
// tile_k_forloop.mlir left the compute generic at rank 3 (Mi, Ni, Ki), each
// extent 32, nested inside the K block-grid scf.for which is itself nested
// inside the M/N scf.forall.
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%module: !transform.any_op {transform.readonly}) {
    %generic = transform.structured.match ops{["linalg.generic"]} in %module
        : (!transform.any_op) -> !transform.any_op
    %tiled_generic, %m_loop, %n_loop, %k_loop =
        transform.structured.tile_using_for %generic tile_sizes [8, 8, 8]
        : (!transform.any_op)
        -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)

    %func = transform.structured.match ops{["func.func"]} attributes{sym_name = "matmul_fn_0"} in %module
        : (!transform.any_op) -> !transform.any_op

    transform.apply_patterns to %func {
      transform.apply_patterns.canonicalization
      transform.apply_patterns.linalg.tiling_canonicalization
    } : !transform.any_op
    transform.apply_cse to %func : !transform.any_op

    transform.yield
  }
}

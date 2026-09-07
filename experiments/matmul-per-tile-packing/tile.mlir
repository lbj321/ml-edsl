// Transform-dialect library: tile the packed matmul's compute generic down
// to vectorization granularity. Runs as its own --transform-interpreter
// pass over out/packed.mlir, so the tiled-but-not-yet-vectorized IR can be
// inspected on its own (out/tiled.mlir).
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%module: !transform.any_op {transform.readonly}) {
    // Tile the packed op's outer block-grid dims (M,N,K) to 1, leaving a
    // 32x32x32 generic per grid cell inside an explicit scf.for nest.
    // Vectorizing the untiled 6D generic directly would fold the
    // block-grid trip counts (4x4x4 here) into the vector shape too, which
    // is wrong - same reasoning as the old linalg-block-pack-matmul
    // experiment's tile_and_vectorize.mlir.
    %generic = transform.structured.match ops{["linalg.generic"]} in %module
        : (!transform.any_op) -> !transform.any_op
    %tiled_generic, %mb_loop, %nb_loop, %kb_loop =
        transform.structured.tile_using_for %generic tile_sizes [1, 1, 1, 0, 0, 0]
        : (!transform.any_op)
        -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)

    %func = transform.structured.match ops{["func.func"]} attributes{sym_name = "matmul"} in %module
        : (!transform.any_op) -> !transform.any_op

    // Tiling-by-one leaves the tiled generic at rank 6 with three
    // unit-extent dims (1x1x1x32x32x32) - fold those away before
    // vectorizing (in the next stage) or linalg::vectorize's
    // contraction-detection won't fire.
    transform.apply_patterns to %func {
      transform.apply_patterns.canonicalization
      transform.apply_patterns.linalg.tiling_canonicalization
    } : !transform.any_op
    transform.apply_cse to %func : !transform.any_op
    transform.apply_patterns to %func {
      transform.apply_patterns.linalg.fold_unit_extent_dims_via_reshapes
    } : !transform.any_op

    // Second-level tile 32x32x32 -> 8x8x8, matching the real compiler's
    // vectorization granularity (LinalgMatmulTilingPass(8,8,8) /
    // LinalgMatmulToContractPass). Vectorizing directly at 32x32x32 works
    // but blows up SSA count (~10x, per the old experiment's measurement).
    %generic_2b = transform.structured.match ops{["linalg.generic"]} in %func
        : (!transform.any_op) -> !transform.any_op
    %tiled_generic_2b, %m_loop, %n_loop, %k_loop =
        transform.structured.tile_using_for %generic_2b tile_sizes [8, 8, 8]
        : (!transform.any_op)
        -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)

    transform.apply_patterns to %func {
      transform.apply_patterns.canonicalization
      transform.apply_patterns.linalg.tiling_canonicalization
    } : !transform.any_op
    transform.apply_cse to %func : !transform.any_op

    transform.yield
  }
}

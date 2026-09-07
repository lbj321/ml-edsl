// Transform-dialect library: tile the packed matmul's compute generic down
// to vectorization granularity. Runs as its own --transform-interpreter
// pass over out/packed.mlir, so the tiled-but-not-yet-vectorized IR can be
// inspected on its own (out/tiled.mlir).
//
// Outer block-grid tiling now mirrors the real compiler's split between
// parallel and serial tiling (LinalgMatmulParallelTilingPass then
// LinalgMatmulKTilingPass): the M/N block-grid dims (parallel iterators) go
// to an scf.forall so the OpenMP conversion path (forall -> parallel ->
// omp.parallel, later in run.sh) has something to act on; the K block-grid
// dim (reduction iterator) stays a serial scf.for, since tiling a reduction
// dim into forall would require per-thread accumulation the plain
// tile_using_forall op doesn't provide - see TileUsingForallOp's own
// "user's responsibility" warning in LinalgTransformOps.td.
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%module: !transform.any_op {transform.readonly}) {
    // Tile the packed op's outer M/N block-grid dims (parallel iterators) to
    // 1 via scf.forall, leaving K block-grid (32) and the 32x32x32 inner
    // tile untouched.
    %generic = transform.structured.match ops{["linalg.generic"]} in %module
        : (!transform.any_op) -> !transform.any_op
    %tiled_generic_0, %forall =
        transform.structured.tile_using_forall %generic tile_sizes [1, 1, 0, 0, 0, 0]
        : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

    %func = transform.structured.match ops{["func.func"]} attributes{sym_name = "matmul"} in %module
        : (!transform.any_op) -> !transform.any_op

    // Tiling-by-one leaves the tiled generic at rank 6 with two unit-extent
    // dims (1x1x32x32x32x32) - fold those away before tiling K block-grid
    // below, same reasoning as the old single-level tile_using_for version.
    transform.apply_patterns to %func {
      transform.apply_patterns.canonicalization
      transform.apply_patterns.linalg.tiling_canonicalization
    } : !transform.any_op
    transform.apply_cse to %func : !transform.any_op
    transform.apply_patterns to %func {
      transform.apply_patterns.linalg.fold_unit_extent_dims_via_reshapes
    } : !transform.any_op

    // Serial K block-grid tiling (32 -> 1), inside the forall body -
    // matches LinalgMatmulKTilingPass running nested inside the outer
    // parallel tile in the real pipeline. Reduction dim, so tile_using_for
    // (iter_args accumulation), not forall.
    %generic_1b = transform.structured.match ops{["linalg.generic"]} in %func
        : (!transform.any_op) -> !transform.any_op
    %tiled_generic_1b, %kb_loop =
        transform.structured.tile_using_for %generic_1b tile_sizes [1, 0, 0, 0]
        : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

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

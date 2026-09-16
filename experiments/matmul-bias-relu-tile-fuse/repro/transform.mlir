// Inner tile+fuse experiment: pull bias_add and relu into the matmul's own
// M/N micro-kernel tiling instead of leaving them as separate 8x8 sibling
// loops produced by LinalgGenericTilingPass. Tile size mirrors
// createLinalgMatmulTilingPass()'s actual (8, 8, 8) -- see
// cpp/src/MLIRLoweringPasses.cpp:571-572 -- not the earlier unverified 4x16
// guess. NOTE: LinalgGenericTilingPass's 8-wide strips exist specifically to
// keep vectorization scoped small enough to avoid an LLVM O3 hang on large
// shapes (combinatorial explosion) -- verify the fused body here still
// vectorizes at 8x8 granularity per tile, not the whole 64x64 outer tile,
// before treating this as a drop-in replacement.
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%module: !transform.any_op {transform.readonly}) {
    %relu = transform.structured.match ops{["linalg.generic"]}
              attributes{library_call = "relu"} in %module
              : (!transform.any_op) -> !transform.any_op

    // Tiles relu to [4, 16] and fuses producers (bias_add, then matmul)
    // back through the resulting loop nest.
    %tiled_relu, %loops:2 = transform.structured.fuse %relu
        {tile_sizes = [8, 8], tile_interchange = [0, 1]}
        : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)

    // fuse only tiled relu's own dims (M, N). The matmul it pulled in still
    // has a full K=128 reduction -- re-match it inside the fused nest and
    // tile just the reduction dim to 8 to restore the 8x8x8 micro-kernel
    // createLinalgMatmulTilingPass() actually uses (cpp/src/MLIRLoweringPasses.cpp:571-572).
    %matmul = transform.structured.match ops{["linalg.matmul"]} in %module
                : (!transform.any_op) -> !transform.any_op

    %tiled_matmul, %k_loop = transform.structured.tile_using_for %matmul tile_sizes [0, 0, 8]
        : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

    transform.yield
  }
}

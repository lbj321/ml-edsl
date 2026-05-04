// GPU two-level tile-and-fuse strategy using the transform dialect.
//
// Level 1 (block): tile relu [32×32] → scf.forall per output block, fuse
//                  bias_add → matmul → fill into that block forall.
// Level 2 (thread): tile the block-level relu [1×1] → scf.forall per
//                   element (= one thread), fuse the block-level ops inward.
//
// After bufferize + scf-forall-to-parallel on both foralls:
//   scf.parallel [4,4]   → blockIdx  (gpu-map-parallel-loops level 0)
//   scf.parallel [32,32] → threadIdx (gpu-map-parallel-loops level 1)
//     scf.for [128]      → sequential K reduction (one per thread)
//
// This produces a single gpu.launch with grid(4,4,1) block(32,32,1), where
// each of the 1024 threads computes one output element of the 32×32 tile.
//
// Run:
//   mlir-opt --transform-interpreter --canonicalize \
//             experiments/tile_fuse_gpu_2level.mlir
//
// Or via the pipeline script (Variant C):
//   ./experiments/tile_fuse_gpu_pipeline.sh

func.func @dense_layer(
    %A:    tensor<128x128xf32>,
    %B:    tensor<128x128xf32>,
    %bias: tensor<128xf32>
) -> tensor<128x128xf32> {

  %cst = arith.constant 0.0 : f32

  %init = tensor.empty() : tensor<128x128xf32>
  %fill = linalg.fill ins(%cst : f32) outs(%init : tensor<128x128xf32>)
        -> tensor<128x128xf32>

  %C = linalg.matmul
    ins(%A, %B   : tensor<128x128xf32>, tensor<128x128xf32>)
    outs(%fill   : tensor<128x128xf32>)
    -> tensor<128x128xf32>

  %init_bias = tensor.empty() : tensor<128x128xf32>
  %biased = linalg.generic {
    indexing_maps = [
      affine_map<(i, j) -> (i, j)>,
      affine_map<(i, j) -> (j)>,
      affine_map<(i, j) -> (i, j)>
    ],
    iterator_types = ["parallel", "parallel"],
    library_call = "bias_add"
  } ins(%C, %bias : tensor<128x128xf32>, tensor<128xf32>)
    outs(%init_bias : tensor<128x128xf32>) {
  ^bb0(%c: f32, %b: f32, %out: f32):
    %sum = arith.addf %c, %b : f32
    linalg.yield %sum : f32
  } -> tensor<128x128xf32>

  %init_relu = tensor.empty() : tensor<128x128xf32>
  %result = linalg.generic {
    indexing_maps = [
      affine_map<(i, j) -> (i, j)>,
      affine_map<(i, j) -> (i, j)>
    ],
    iterator_types = ["parallel", "parallel"],
    library_call = "relu"
  } ins(%biased : tensor<128x128xf32>)
    outs(%init_relu : tensor<128x128xf32>) {
  ^bb0(%c: f32, %out: f32):
    %r = arith.maximumf %c, %cst : f32
    linalg.yield %r : f32
  } -> tensor<128x128xf32>

  return %result : tensor<128x128xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root : !transform.any_op {transform.readonly}) {
    %relu   = transform.structured.match attributes {library_call = "relu"} in %root
                : (!transform.any_op) -> !transform.any_op
    %bias   = transform.structured.match attributes {library_call = "bias_add"} in %root
                : (!transform.any_op) -> !transform.any_op
    %matmul = transform.structured.match ops{["linalg.matmul"]} in %root
                : (!transform.any_op) -> !transform.any_op
    %fill   = transform.structured.match ops{["linalg.fill"]} in %root
                : (!transform.any_op) -> !transform.any_op

    // Level 1 — block tiling: one scf.forall tile per GPU thread block.
    %tiled_relu, %block_forall =
        transform.structured.tile_using_forall %relu tile_sizes [32, 32]
            : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

    // Fuse producers into the block forall; capture the block-level handles
    // so we can fuse them into the thread forall in level 2.
    %block_bias, %block_forall2 =
        transform.structured.fuse_into_containing_op %bias into %block_forall
            : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    %block_matmul, %block_forall3 =
        transform.structured.fuse_into_containing_op %matmul into %block_forall2
            : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    %block_fill, %block_forall4 =
        transform.structured.fuse_into_containing_op %fill into %block_forall3
            : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)

    // Level 2 — thread tiling: one element per thread (1×1 tiles).
    // Tiles the block-level relu — each thread computes exactly one output element.
    %thread_relu, %thread_forall =
        transform.structured.tile_using_forall %tiled_relu tile_sizes [1, 1]
            : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

    // Fuse the block-level ops into the thread forall.
    // Each is sliced to [1×1] (element-wise) or [1×K×1] (matmul, K stays sequential).
    transform.structured.fuse_into_containing_op %block_bias into %thread_forall
        : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.structured.fuse_into_containing_op %block_matmul into %thread_forall
        : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.structured.fuse_into_containing_op %block_fill into %thread_forall
        : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)

    transform.yield
  }
}

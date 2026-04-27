// GPU tile-and-fuse strategy using transform dialect.
//
// Tiles relu [32×32] into scf.forall with GPU block mapping, then fuses
// bias_add → matmul → fill into the same forall body. All in tensor land
// before bufferization. The single forall outlines to ONE gpu.launch kernel
// (fill → matmul → bias_add → relu per block), replacing the current
// two-operation approach (separate matmul kernel + bias+relu kernel).
//
// Run:
//   mlir-opt --transform-interpreter --cse --canonicalize \
//             experiments/tile_fuse_gpu_strategy.mlir
//
// Expected: scf.forall [4, 4] body with gpu.block mapping contains
//   linalg.fill → linalg.matmul → linalg.generic(bias_add) → linalg.generic(relu)

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

    // Tile relu [32×32] with GPU block mapping — each tile = one thread block
    // Results: (tiled_op, forall_op)
    %tiled_relu, %forall =
        transform.structured.tile_using_forall %relu tile_sizes [32, 32]
            (mapping = [#gpu.block<y>, #gpu.block<x>])
            : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

    // Fuse producers inward: bias_add → matmul → fill
    transform.structured.fuse_into_containing_op %bias into %forall
        : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.structured.fuse_into_containing_op %matmul into %forall
        : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.structured.fuse_into_containing_op %fill into %forall
        : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)

    transform.yield
  }
}

// Test: void + writable out-param with materialize_in_destination.
//
// %out is marked bufferization.writable so one-shot-bufferize treats its
// buffer as pre-allocated and writes the entire fill→matmul→bias→relu chain
// into it in-place. materialize_in_destination asserts the result lands in
// %out — no memref.copy needed.
//
// Test bufferization only (check for absence of memref.copy):
//   mlir-opt \
//     "--one-shot-bufferize=bufferize-function-boundaries=true function-boundary-type-conversion=identity-layout-map" \
//     --canonicalize \
//     experiments/outparam_test.mlir | grep "memref\.copy"
//
// Full pipeline through GPU outlining:
//   mlir-opt \
//     --transform-interpreter --canonicalize \
//     "--one-shot-bufferize=bufferize-function-boundaries=true function-boundary-type-conversion=identity-layout-map" \
//     --promote-buffers-to-stack --canonicalize \
//     --scf-forall-to-parallel \
//     --convert-linalg-to-loops \
//     --gpu-map-parallel-loops \
//     --convert-parallel-loops-to-gpu \
//     --gpu-kernel-outlining --canonicalize \
//     experiments/outparam_test.mlir

func.func @dense_layer(
    %A:    tensor<128x128xf32>,
    %B:    tensor<128x128xf32>,
    %bias: tensor<128xf32>,
    %out:  tensor<128x128xf32> {bufferization.writable = true}
) -> tensor<128x128xf32> {
  %cst = arith.constant 0.0 : f32

  %filled = linalg.fill ins(%cst : f32) outs(%out : tensor<128x128xf32>)
          -> tensor<128x128xf32>

  %C = linalg.matmul
      ins(%A, %B : tensor<128x128xf32>, tensor<128x128xf32>)
      outs(%filled : tensor<128x128xf32>)
      -> tensor<128x128xf32>

  %buf1 = tensor.empty() : tensor<128x128xf32>
  %biased = linalg.generic {
    indexing_maps = [
      affine_map<(i, j) -> (i, j)>,
      affine_map<(i, j) -> (j)>,
      affine_map<(i, j) -> (i, j)>
    ],
    iterator_types = ["parallel", "parallel"],
    library_call = "bias_add"
  } ins(%C, %bias : tensor<128x128xf32>, tensor<128xf32>)
    outs(%buf1 : tensor<128x128xf32>) {
  ^bb0(%c: f32, %b: f32, %o: f32):
    %sum = arith.addf %c, %b : f32
    linalg.yield %sum : f32
  } -> tensor<128x128xf32>

  %buf2 = tensor.empty() : tensor<128x128xf32>
  %result = linalg.generic {
    indexing_maps = [
      affine_map<(i, j) -> (i, j)>,
      affine_map<(i, j) -> (i, j)>
    ],
    iterator_types = ["parallel", "parallel"],
    library_call = "relu"
  } ins(%biased : tensor<128x128xf32>)
    outs(%buf2 : tensor<128x128xf32>) {
  ^bb0(%c: f32, %o: f32):
    %r = arith.maximumf %c, %cst : f32
    linalg.yield %r : f32
  } -> tensor<128x128xf32>

  return %result : tensor<128x128xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root : !transform.any_op {transform.readonly}) {
    %relu   = transform.structured.match attributes {library_call = "relu"}     in %root : (!transform.any_op) -> !transform.any_op
    %bias   = transform.structured.match attributes {library_call = "bias_add"} in %root : (!transform.any_op) -> !transform.any_op
    %matmul = transform.structured.match ops{["linalg.matmul"]}                 in %root : (!transform.any_op) -> !transform.any_op
    %fill   = transform.structured.match ops{["linalg.fill"]}                   in %root : (!transform.any_op) -> !transform.any_op

    %tiled_relu, %forall =
        transform.structured.tile_using_forall %relu tile_sizes [32, 32]
            : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.structured.fuse_into_containing_op %bias   into %forall : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.structured.fuse_into_containing_op %matmul into %forall : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.structured.fuse_into_containing_op %fill   into %forall : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.yield
  }
}

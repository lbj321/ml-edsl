// dense_gpu.mlir
//
// Run:
//   mlir-opt dense_gpu.mlir \
//     --transform-interpreter \
//     --test-transform-dialect-erase-schedule \
//     --canonicalize -cse
//
// NOTE: payload IR and transform IR are in the SAME module so that
// transform-interpreter can find both the func and the named sequence.

#map_2d   = affine_map<(d0, d1) -> (d0, d1)>
#map_bias = affine_map<(d0, d1) -> (d1)>

module attributes {transform.with_named_sequence} {

  // -----------------------------------------------------------------------
  // Payload IR: matmul + bias_add + relu on 256x256 f32
  // -----------------------------------------------------------------------
  func.func @dense(
      %A:    tensor<256x256xf32>,
      %B:    tensor<256x256xf32>,
      %bias: tensor<256xf32>
  ) -> tensor<256x256xf32> {

    %cst  = arith.constant 0.000000e+00 : f32
    %acc  = tensor.empty() : tensor<256x256xf32>
    %zero = linalg.fill ins(%cst : f32)
                        outs(%acc : tensor<256x256xf32>)
            -> tensor<256x256xf32>

    %mm = linalg.matmul
            ins(%A, %B : tensor<256x256xf32>, tensor<256x256xf32>)
            outs(%zero : tensor<256x256xf32>)
          -> tensor<256x256xf32>

    %bias_out = tensor.empty() : tensor<256x256xf32>
    %biased = linalg.generic {
        indexing_maps  = [#map_2d, #map_bias, #map_2d],
        iterator_types = ["parallel", "parallel"],
        library_call   = "bias_add"
      }
      ins(%mm, %bias : tensor<256x256xf32>, tensor<256xf32>)
      outs(%bias_out : tensor<256x256xf32>) {
      ^bb0(%in: f32, %b: f32, %out: f32):
        %add = arith.addf %in, %b : f32
        linalg.yield %add : f32
    } -> tensor<256x256xf32>

    %relu_out = tensor.empty() : tensor<256x256xf32>
    %relued = linalg.generic {
        indexing_maps  = [#map_2d, #map_2d],
        iterator_types = ["parallel", "parallel"],
        library_call   = "relu"
      }
      ins(%biased : tensor<256x256xf32>)
      outs(%relu_out : tensor<256x256xf32>) {
      ^bb0(%in: f32, %out: f32):
        %zero_f = arith.constant 0.000000e+00 : f32
        %max    = arith.maximumf %in, %zero_f : f32
        linalg.yield %max : f32
    } -> tensor<256x256xf32>

    return %relued : tensor<256x256xf32>
  }

  // -----------------------------------------------------------------------
  // Transform IR
  //
  // Two-level tiling without GPU mapping attributes.
  // Add  mapping [#gpu.block<y>, #gpu.block<x>]  /
  //      mapping [#gpu.thread<y>, #gpu.thread<x>]
  // to the tile_using_forall calls once you confirm fusion works.
  // -----------------------------------------------------------------------
  transform.named_sequence @__transform_main(
      %root : !transform.any_op {transform.readonly}) {

    %relu = transform.structured.match
        attributes {library_call = "relu"} in %root
        : (!transform.any_op) -> !transform.any_op

    %bias = transform.structured.match
        attributes {library_call = "bias_add"} in %root
        : (!transform.any_op) -> !transform.any_op

    %matmul = transform.structured.match
        ops{["linalg.matmul"]} in %root
        : (!transform.any_op) -> !transform.any_op

    %fill = transform.structured.match
        ops{["linalg.fill"]} in %root
        : (!transform.any_op) -> !transform.any_op

    // Level 1 — block tiling (32x32 tiles, 8x8 grid over 256x256)
    %tiled_relu, %block_forall =
        transform.structured.tile_using_forall %relu tile_sizes [32, 32]
            : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

    %block_bias, %block_forall_2 =
        transform.structured.fuse_into_containing_op %bias into %block_forall
            : (!transform.any_op, !transform.any_op)
           -> (!transform.any_op, !transform.any_op)

    %block_matmul, %block_forall_3 =
        transform.structured.fuse_into_containing_op %matmul into %block_forall_2
            : (!transform.any_op, !transform.any_op)
           -> (!transform.any_op, !transform.any_op)

    %block_fill, %block_forall_4 =
        transform.structured.fuse_into_containing_op %fill into %block_forall_3
            : (!transform.any_op, !transform.any_op)
           -> (!transform.any_op, !transform.any_op)

    // Level 2 — thread tiling (1x1 tiles, 32x32 threads per block)
    %thread_relu, %thread_forall =
        transform.structured.tile_using_forall %tiled_relu tile_sizes [1, 1]
            : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

    transform.structured.fuse_into_containing_op %block_bias into %thread_forall
        : (!transform.any_op, !transform.any_op)
       -> (!transform.any_op, !transform.any_op)

    transform.structured.fuse_into_containing_op %block_matmul into %thread_forall
        : (!transform.any_op, !transform.any_op)
       -> (!transform.any_op, !transform.any_op)

    transform.structured.fuse_into_containing_op %block_fill into %thread_forall
        : (!transform.any_op, !transform.any_op)
       -> (!transform.any_op, !transform.any_op)

    transform.yield
  }

}

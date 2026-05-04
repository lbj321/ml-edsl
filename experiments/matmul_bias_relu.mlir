// matmul + bias add + relu in tensor land (functional style).
//
// Bias add and relu use the ins/outs split pattern so that producer-consumer
// chains are explicit and the transform dialect fuse op can walk them.
//
// Run via fuse_matmul_bias_relu.sh, or directly:
//   mlir-opt --linalg-fuse-elementwise-ops --transform-interpreter \
//             --cse --canonicalize experiments/matmul_bias_relu.mlir
//
// RUN: mlir-opt --linalg-fuse-elementwise-ops --transform-interpreter \
// RUN:           --cse --canonicalize %s | \
// RUN:   FileCheck %s
//
// CHECK-LABEL: func.func @matmul_bias_relu
// CHECK-DAG:    %[[C16:.*]] = arith.constant 16
// CHECK:        scf.for %{{.*}} = %{{.*}} to %{{.*}} step %[[C16]]
// CHECK:          scf.for %{{.*}} = %{{.*}} to %{{.*}} step %[[C16]]
// CHECK:            linalg.fill
// CHECK:            linalg.matmul
// CHECK:            linalg.generic

func.func @matmul_bias_relu(
    %A:    tensor<64x64xf32>,
    %B:    tensor<64x64xf32>,
    %bias: tensor<64xf32>
) -> tensor<64x64xf32> {

  %cst = arith.constant 0.0 : f32

  %init = tensor.empty() : tensor<64x64xf32>
  %fill = linalg.fill ins(%cst : f32) outs(%init : tensor<64x64xf32>)
        -> tensor<64x64xf32>

  // matmul: C = A * B  (accumulates into zero-filled tensor)
  %C = linalg.matmul
    ins(%A, %B   : tensor<64x64xf32>, tensor<64x64xf32>)
    outs(%fill   : tensor<64x64xf32>)
    -> tensor<64x64xf32>

  %init_bias = tensor.empty() : tensor<64x64xf32>

  // bias add: out[i,j] = C[i,j] + bias[j]
  %biased = linalg.generic {
    indexing_maps = [
      affine_map<(i, j) -> (i, j)>,  // C        (ins)
      affine_map<(i, j) -> (j)>,      // bias     (ins, broadcast over i)
      affine_map<(i, j) -> (i, j)>   // out      (outs)
    ],
    iterator_types = ["parallel", "parallel"]
  } ins(%C, %bias : tensor<64x64xf32>, tensor<64xf32>)
    outs(%init_bias : tensor<64x64xf32>) {
  ^bb0(%c: f32, %b: f32, %out: f32):
    %sum = arith.addf %c, %b : f32
    linalg.yield %sum : f32
  } -> tensor<64x64xf32>

  %init_relu = tensor.empty() : tensor<64x64xf32>

  // relu: out[i,j] = max(biased[i,j], 0)
  %result = linalg.generic {
    indexing_maps = [
      affine_map<(i, j) -> (i, j)>,  // biased   (ins)
      affine_map<(i, j) -> (i, j)>   // out      (outs)
    ],
    iterator_types = ["parallel", "parallel"]
  } ins(%biased : tensor<64x64xf32>)
    outs(%init_relu : tensor<64x64xf32>) {
  ^bb0(%c: f32, %out: f32):
    %r = arith.maximumf %c, %cst : f32
    linalg.yield %r : f32
  } -> tensor<64x64xf32>

  return %result : tensor<64x64xf32>
}

// Transform sequence:
//   --linalg-fuse-elementwise-ops (runs first) merges bias_add + relu into one
//   linalg.generic. The transform sequence then sees exactly one generic and
//   tiles it [16, 16], greedily fusing fill + matmul upward:
//      merged_generic ← matmul ← fill
//   Result: a single scf.for nest containing all ops on 16x16 tiles.
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root : !transform.any_op {transform.readonly}) {
    // bias_add and relu are already merged into one generic by
    // --linalg-fuse-elementwise-ops before the interpreter runs.
    %merged = transform.structured.match ops{["linalg.generic"]} in %root
      : (!transform.any_op) -> !transform.any_op

    %tiled, %loop_i, %loop_j = transform.structured.fuse %merged [16, 16]
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)

    transform.yield
  }
}

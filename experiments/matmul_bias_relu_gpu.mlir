// matmul + bias add + relu — GPU single-level schedule (128×128).
//
// Mirrors the C++ GPU pipeline in MLIRLowering.cpp:
//   LinalgGPUMatmulTilingPass — 32×32 tiles, K left untiled.
//   Each 32×32 tile maps to one GPU thread block (32×32 = 1024 threads, CUDA max).
//   K is untiled: each tile block accumulates the full K=128 reduction in one pass,
//   keeping the partial sum in shared memory / registers.
//
// Note: GPU mapping (scf.forall → gpu.launch) and NVVM/PTX lowering require
// post-bufferization passes not available in tensor-land experiments.
// This file validates the tiling + fusion structure only.
//
// Pass ordering: elementwise fusion runs before the transform interpreter so
// bias_add + relu are pre-merged into one linalg.generic.
//
// Run via fuse_matmul_bias_relu.sh, or directly:
//   mlir-opt --linalg-fuse-elementwise-ops --transform-interpreter \
//             --cse --canonicalize experiments/matmul_bias_relu_gpu.mlir
//
// RUN: mlir-opt --linalg-fuse-elementwise-ops --transform-interpreter \
// RUN:           --cse --canonicalize %s | \
// RUN:   FileCheck %s
//
// CHECK-LABEL: func.func @matmul_bias_relu_gpu
// CHECK-DAG:    %[[C32:.*]] = arith.constant 32
// CHECK:        scf.for %{{.*}} = %{{.*}} to %{{.*}} step %[[C32]]
// CHECK:          scf.for %{{.*}} = %{{.*}} to %{{.*}} step %[[C32]]
// CHECK:            linalg.fill
// CHECK:            linalg.matmul
// CHECK:            linalg.generic

func.func @matmul_bias_relu_gpu(
    %A:    tensor<128x128xf32>,
    %B:    tensor<128x128xf32>,
    %bias: tensor<128xf32>
) -> tensor<128x128xf32> {

  %cst = arith.constant 0.0 : f32

  %init = tensor.empty() : tensor<128x128xf32>
  %fill = linalg.fill ins(%cst : f32) outs(%init : tensor<128x128xf32>)
        -> tensor<128x128xf32>

  // matmul: C = A * B
  %C = linalg.matmul
    ins(%A, %B   : tensor<128x128xf32>, tensor<128x128xf32>)
    outs(%fill   : tensor<128x128xf32>)
    -> tensor<128x128xf32>

  %init_bias = tensor.empty() : tensor<128x128xf32>

  // bias add: out[i,j] = C[i,j] + bias[j]
  %biased = linalg.generic {
    indexing_maps = [
      affine_map<(i, j) -> (i, j)>,
      affine_map<(i, j) -> (j)>,
      affine_map<(i, j) -> (i, j)>
    ],
    iterator_types = ["parallel", "parallel"]
  } ins(%C, %bias : tensor<128x128xf32>, tensor<128xf32>)
    outs(%init_bias : tensor<128x128xf32>) {
  ^bb0(%c: f32, %b: f32, %out: f32):
    %sum = arith.addf %c, %b : f32
    linalg.yield %sum : f32
  } -> tensor<128x128xf32>

  %init_relu = tensor.empty() : tensor<128x128xf32>

  // relu: out[i,j] = max(biased[i,j], 0)
  %result = linalg.generic {
    indexing_maps = [
      affine_map<(i, j) -> (i, j)>,
      affine_map<(i, j) -> (i, j)>
    ],
    iterator_types = ["parallel", "parallel"]
  } ins(%biased : tensor<128x128xf32>)
    outs(%init_relu : tensor<128x128xf32>) {
  ^bb0(%c: f32, %out: f32):
    %r = arith.maximumf %c, %cst : f32
    linalg.yield %r : f32
  } -> tensor<128x128xf32>

  return %result : tensor<128x128xf32>
}

// Transform sequence (single level, K untiled):
//   --linalg-fuse-elementwise-ops pre-merges bias_add + relu → 1 generic.
//   Tile merged generic [32, 32], fuse fill + matmul upward.
//   K is untiled (tile_size = 0 implicitly) — each 32×32 tile accumulates
//   the full K=128 reduction, matching the GPU thread-block tiling strategy.
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root : !transform.any_op {transform.readonly}) {
    // bias_add + relu already merged into one generic by --linalg-fuse-elementwise-ops.
    %merged = transform.structured.match ops{["linalg.generic"]} in %root
      : (!transform.any_op) -> !transform.any_op

    // Single level: 32×32 tile grid (4×4 = 16 tiles), K untiled.
    %tiled, %loop_i, %loop_j = transform.structured.fuse %merged [32, 32]
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)

    transform.yield
  }
}

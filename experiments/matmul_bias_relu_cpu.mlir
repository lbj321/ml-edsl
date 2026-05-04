// matmul + bias add + relu — CPU two-level hierarchical schedule (128×128).
//
// Mirrors the C++ CPU pipeline in MLIRLowering.cpp:
//   Level 1: LinalgMatmulParallelTilingPass — 64×64 outer tiles (→ scf.forall → omp.parallel)
//   Level 2: LinalgMatmulTilingPass         — 8×8×8 inner tiles (→ scf.for, vectorizable)
//
// Pass ordering: elementwise fusion runs before the transform interpreter so
// bias_add + relu are pre-merged into one linalg.generic. The transform
// sequence then sees exactly one generic and two-level-tiles the whole chain.
//
// Run via fuse_matmul_bias_relu.sh, or directly:
//   mlir-opt --linalg-fuse-elementwise-ops --transform-interpreter \
//             --cse --canonicalize experiments/matmul_bias_relu_cpu.mlir
//
// RUN: mlir-opt --linalg-fuse-elementwise-ops --transform-interpreter \
// RUN:           --cse --canonicalize %s | \
// RUN:   FileCheck %s
//
// CHECK-LABEL: func.func @matmul_bias_relu_cpu
// CHECK-DAG:    %[[C64:.*]] = arith.constant 64
// CHECK:        scf.for %{{.*}} = %{{.*}} to %{{.*}} step %[[C64]]
// CHECK:          scf.for %{{.*}} = %{{.*}} to %{{.*}} step %[[C64]]
// CHECK:            linalg.fill
// CHECK:            scf.for
// CHECK:              scf.for
// CHECK:                scf.for
// CHECK:                  linalg.matmul
// CHECK:            linalg.generic

func.func @matmul_bias_relu_cpu(
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

// Transform sequence (two-level hierarchical):
//   --linalg-fuse-elementwise-ops pre-merges bias_add + relu → 1 generic.
//   Level 1: fuse merged generic [64, 64] — outer tile grid (2×2 = 4 tiles).
//            fill + matmul are fused upward into each outer tile.
//   Level 2: tile_using_for matmul [8, 8, 8] — inner M/N/K loops over the
//            64×64×128 matmul tile, producing 8×8×8 microkernel blocks.
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root : !transform.any_op {transform.readonly}) {
    // bias_add + relu already merged into one generic by --linalg-fuse-elementwise-ops.
    %merged = transform.structured.match ops{["linalg.generic"]} in %root
      : (!transform.any_op) -> !transform.any_op

    // Level 1: outer 64×64 tile grid, fuse fill + matmul upward.
    %tiled, %loop_i, %loop_j = transform.structured.fuse %merged [64, 64]
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)

    // Level 2: find the matmul inside the outer loop region and tile [8, 8, 8].
    %matmul = transform.structured.match ops{["linalg.matmul"]} in %loop_i
      : (!transform.any_op) -> !transform.any_op
    %tiled_mm, %loops:3 = transform.structured.tile_using_for %matmul tile_sizes [8, 8, 8]
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op,
                                 !transform.any_op, !transform.any_op)

    transform.yield
  }
}

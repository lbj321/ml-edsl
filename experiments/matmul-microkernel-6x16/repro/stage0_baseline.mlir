// Stage 0 baseline: plain linalg.matmul through the current production
// pipeline (mlir-edsl-opt -cpu-pipeline == buildCPUPipeline in
// cpp/src/MLIRLowering.cpp).
//
// Sized 192x256x192 (M=N=192, K=256). The production pipeline's outer
// tiling (LinalgOuterTileAndFusePass / LinalgMatmulParallelTilingPass) uses
// a hardcoded 64x64 tile with no remainder padding yet, so M and N must
// also divide 64 for a clean steady-state run -- 336 (lcm(6,16,168)) does
// NOT divide 64 and hits linalg-vectorize's un-vectorized fallback path on
// the tail tile (confirmed: `mlir-edsl-opt -cpu-pipeline` on 336x256x336
// emits "vectorization failed, skipping op" for a dynamic ?x8 remainder
// tile). 192 divides 6, 16, and 64 (not 168/256 -- that's fine, this input
// is only used for the Stage 0 production-pipeline baseline, not fed
// through the Stage 1+ custom blocking). Bump to a larger 64-multiple (e.g.
// 1024x1024x1024) once the harness is validated, for a size large enough
// to actually stress cache blocking.
//
// Mirrors the tensor-semantics shape the EDSL frontend actually emits
// (bufferization.to_tensor/materialize_in_destination at the memref
// boundary, linalg.fill zeroing the accumulator) -- see
// experiments/matmul-bias-relu-tile-fuse/repro/input.mlir for the precedent.
module {
  func.func @matmul_baseline(%arg0: memref<192x256xf32>, %arg1: memref<256x192xf32>, %arg2: memref<192x192xf32>) {
    %0 = bufferization.to_tensor %arg0 restrict : memref<192x256xf32> to tensor<192x256xf32>
    %1 = bufferization.to_tensor %arg1 restrict : memref<256x192xf32> to tensor<256x192xf32>
    %cst = arith.constant 0.000000e+00 : f32
    %2 = tensor.empty() : tensor<192x192xf32>
    %3 = linalg.fill ins(%cst : f32) outs(%2 : tensor<192x192xf32>) -> tensor<192x192xf32>
    %4 = linalg.matmul ins(%0, %1 : tensor<192x256xf32>, tensor<256x192xf32>) outs(%3 : tensor<192x192xf32>) -> tensor<192x192xf32>
    bufferization.materialize_in_destination %4 in restrict writable %arg2 : (tensor<192x192xf32>, memref<192x192xf32>) -> ()
    return
  }
}

// Stage 2: the macro-kernel -- Stage 1's 6x16xKC microkernel wrapped in the
// jr (N register-block) and ir (M register-block) loops, matching BLIS's
// loop order jr (outer) -> ir (inner) -> kernel (see PLAN.md Stage 3's full
// loop order: jc -> pc -> ic -> jr -> ir -> kernel; Stage 2 only exercises
// the innermost two).
//
// A and B are already-packed panels passed in directly (no packing yet --
// that's Stage 3): A is KC x MC k-major (A[k,m], same layout as Stage 1,
// just MC=168 columns instead of 6), B is KC x NC (B[k,n], NC=64 here).
// MC=168 and KC=256 match PLAN.md's Stage 2 done-when check; NC=64 is just
// large enough to exercise multiple jr iterations (4 tiles of NR=16).
func.func @macrokernel_168x64(%A: memref<256x168xf32>, %B: memref<256x64xf32>, %C: memref<168x64xf32>) {
  %a = bufferization.to_tensor %A restrict : memref<256x168xf32> to tensor<256x168xf32>
  %b = bufferization.to_tensor %B restrict : memref<256x64xf32> to tensor<256x64xf32>
  %cst = arith.constant 0.000000e+00 : f32
  %init = tensor.empty() : tensor<168x64xf32>
  %filled = linalg.fill ins(%cst : f32) outs(%init : tensor<168x64xf32>) -> tensor<168x64xf32>
  %res = linalg.matmul_transpose_a ins(%a, %b : tensor<256x168xf32>, tensor<256x64xf32>) outs(%filled : tensor<168x64xf32>) -> tensor<168x64xf32>
  bufferization.materialize_in_destination %res in restrict writable %C : (tensor<168x64xf32>, memref<168x64xf32>) -> ()
  return
}

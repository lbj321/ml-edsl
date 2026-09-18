// Stage 1: the 6x16 microkernel in isolation, KC=256.
//
// A is packed KC x MR (k-major: A[k, m]), B is packed KC x NR (B[k, n]) --
// the BLIS packed-panel layout. linalg.matmul_transpose_a gives exactly
// this contraction (ins A: KxM, B: KxN, outs C: MxN -- verified against
// the pinned LLVM 21 build directly; its LinalgNamedStructuredOps.yaml
// shape_map comments read confusingly transposed, but the actual op
// interface and generated indexing_maps are A[k,m], B[k,n], C[m,n]).
//
// KC=256 here is just the k-loop trip count for this isolated kernel test
// (no divisibility constraint vs. MR/NR -- that only matters once M/N
// exceed one tile, at Stage 2+).
func.func @microkernel_6x16(%A: memref<256x6xf32>, %B: memref<256x16xf32>, %C: memref<6x16xf32>) {
  %a = bufferization.to_tensor %A restrict : memref<256x6xf32> to tensor<256x6xf32>
  %b = bufferization.to_tensor %B restrict : memref<256x16xf32> to tensor<256x16xf32>
  %cst = arith.constant 0.000000e+00 : f32
  %init = tensor.empty() : tensor<6x16xf32>
  %filled = linalg.fill ins(%cst : f32) outs(%init : tensor<6x16xf32>) -> tensor<6x16xf32>
  %res = linalg.matmul_transpose_a ins(%a, %b : tensor<256x6xf32>, tensor<256x16xf32>) outs(%filled : tensor<6x16xf32>) -> tensor<6x16xf32>
  bufferization.materialize_in_destination %res in restrict writable %C : (tensor<6x16xf32>, memref<6x16xf32>) -> ()
  return
}

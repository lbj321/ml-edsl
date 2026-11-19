// Stage 3, approach 1 (global pack first): the full (unpacked) problem, at a
// size that exercises every outer loop level with a small iteration count
// for fast iteration: M=336 (2 ic blocks of MC=168), N=64 (2 jc blocks of
// NC=32, each 2 jr of NR=16), K=512 (2 pc blocks of KC=256).
func.func @outer_336x64x512(%A: memref<336x512xf32>, %B: memref<512x64xf32>, %C: memref<336x64xf32>) {
  %a = bufferization.to_tensor %A restrict : memref<336x512xf32> to tensor<336x512xf32>
  %b = bufferization.to_tensor %B restrict : memref<512x64xf32> to tensor<512x64xf32>
  %cst = arith.constant 0.000000e+00 : f32
  %init = tensor.empty() : tensor<336x64xf32>
  %filled = linalg.fill ins(%cst : f32) outs(%init : tensor<336x64xf32>) -> tensor<336x64xf32>
  %res = linalg.matmul ins(%a, %b : tensor<336x512xf32>, tensor<512x64xf32>) outs(%filled : tensor<336x64xf32>) -> tensor<336x64xf32>
  bufferization.materialize_in_destination %res in restrict writable %C : (tensor<336x64xf32>, memref<336x64xf32>) -> ()
  return
}

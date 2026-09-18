// Stage 3 large problem: M=1008 (6 ic blocks of MC=168), N=1024 (4 jc blocks of NC=256), K=1024 (4 pc blocks of KC=256).
func.func @outer_1008x1024x1024(%A: memref<1008x1024xf32>, %B: memref<1024x1024xf32>, %C: memref<1008x1024xf32>) {
  %a = bufferization.to_tensor %A restrict : memref<1008x1024xf32> to tensor<1008x1024xf32>
  %b = bufferization.to_tensor %B restrict : memref<1024x1024xf32> to tensor<1024x1024xf32>
  %cst = arith.constant 0.000000e+00 : f32
  %init = tensor.empty() : tensor<1008x1024xf32>
  %filled = linalg.fill ins(%cst : f32) outs(%init : tensor<1008x1024xf32>) -> tensor<1008x1024xf32>
  %res = linalg.matmul ins(%a, %b : tensor<1008x1024xf32>, tensor<1024x1024xf32>) outs(%filled : tensor<1008x1024xf32>) -> tensor<1008x1024xf32>
  bufferization.materialize_in_destination %res in restrict writable %C : (tensor<1008x1024xf32>, memref<1008x1024xf32>) -> ()
  return
}

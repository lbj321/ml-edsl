// Matches the real compiler's emitted IR exactly (see
// cpp/src/builders/LinalgBuilder.cpp::buildMatmul + MLIRCompiler.cpp's
// createFunction/finalizeFunction): memref args, to_tensor restrict on
// inputs, a fresh tensor.empty+linalg.fill(0) init (never reads %C, so no
// RaW conflict at materialize_in_destination), linalg.matmul, then
// materialize_in_destination into the memref out-param, void return. C is
// a pure out-param here - not an input accumulator - which is what the
// frontend actually emits.
func.func @matmul(%A: memref<1024x1024xf32>, %B: memref<1024x1024xf32>, %C: memref<1024x1024xf32>) {
  %tA = bufferization.to_tensor %A restrict : memref<1024x1024xf32> to tensor<1024x1024xf32>
  %tB = bufferization.to_tensor %B restrict : memref<1024x1024xf32> to tensor<1024x1024xf32>
  %cst = arith.constant 0.000000e+00 : f32
  %init = tensor.empty() : tensor<1024x1024xf32>
  %filled = linalg.fill ins(%cst : f32) outs(%init : tensor<1024x1024xf32>) -> tensor<1024x1024xf32>
  %0 = linalg.matmul ins(%tA, %tB : tensor<1024x1024xf32>, tensor<1024x1024xf32>)
                      outs(%filled : tensor<1024x1024xf32>) -> tensor<1024x1024xf32>
  bufferization.materialize_in_destination %0 in restrict writable %C
      : (tensor<1024x1024xf32>, memref<1024x1024xf32>) -> ()
  return
}

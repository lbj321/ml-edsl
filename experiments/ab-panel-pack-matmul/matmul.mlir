module {
  func.func @matmul_fn_0(%arg0: memref<1024x1024xf32>, %arg1: memref<1024x1024xf32>, %arg2: memref<1024x1024xf32>) {
    %0 = bufferization.to_tensor %arg0 restrict : memref<1024x1024xf32> to tensor<1024x1024xf32>
    %1 = bufferization.to_tensor %arg1 restrict : memref<1024x1024xf32> to tensor<1024x1024xf32>
    %cst = arith.constant 0.000000e+00 : f32
    %2 = tensor.empty() : tensor<1024x1024xf32>
    %3 = linalg.fill ins(%cst : f32) outs(%2 : tensor<1024x1024xf32>) -> tensor<1024x1024xf32>
    %4 = linalg.matmul ins(%0, %1 : tensor<1024x1024xf32>, tensor<1024x1024xf32>) outs(%3 : tensor<1024x1024xf32>) -> tensor<1024x1024xf32>
    bufferization.materialize_in_destination %4 in restrict writable %arg2 : (tensor<1024x1024xf32>, memref<1024x1024xf32>) -> ()
    return
  }
}

#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d1)>
module {
  func.func @dense_512(%arg0: memref<512x512xf32>, %arg1: memref<512x512xf32>, %arg2: memref<512xf32>, %arg3: memref<512x512xf32>) {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = bufferization.to_tensor %arg0 restrict : memref<512x512xf32> to tensor<512x512xf32>
    %1 = bufferization.to_tensor %arg1 restrict : memref<512x512xf32> to tensor<512x512xf32>
    %2 = bufferization.to_tensor %arg2 restrict : memref<512xf32> to tensor<512xf32>
    %3 = tensor.empty() : tensor<512x512xf32>
    %4 = tensor.empty() : tensor<512x512xf32>
    %5 = tensor.empty() : tensor<512x512xf32>
    %6 = scf.forall (%arg4, %arg5) = (0, 0) to (512, 512) step (64, 64) shared_outs(%arg6 = %5) -> (tensor<512x512xf32>) {
      %extracted_slice = tensor.extract_slice %0[%arg4, 0] [64, 512] [1, 1] : tensor<512x512xf32> to tensor<64x512xf32>
      %extracted_slice_0 = tensor.extract_slice %1[0, %arg5] [512, 64] [1, 1] : tensor<512x512xf32> to tensor<512x64xf32>
      %extracted_slice_1 = tensor.extract_slice %4[%arg4, %arg5] [64, 64] [1, 1] : tensor<512x512xf32> to tensor<64x64xf32>
      %7 = linalg.fill ins(%cst : f32) outs(%extracted_slice_1 : tensor<64x64xf32>) -> tensor<64x64xf32>
      %8 = linalg.matmul ins(%extracted_slice, %extracted_slice_0 : tensor<64x512xf32>, tensor<512x64xf32>) outs(%7 : tensor<64x64xf32>) -> tensor<64x64xf32>
      %extracted_slice_2 = tensor.extract_slice %2[%arg5] [64] [1] : tensor<512xf32> to tensor<64xf32>
      %extracted_slice_3 = tensor.extract_slice %3[%arg4, %arg5] [64, 64] [1, 1] : tensor<512x512xf32> to tensor<64x64xf32>
      %9 = linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel", "parallel"], library_call = "bias_add"} ins(%8, %extracted_slice_2 : tensor<64x64xf32>, tensor<64xf32>) outs(%extracted_slice_3 : tensor<64x64xf32>) {
      ^bb0(%in: f32, %in_5: f32, %out: f32):
        %11 = arith.addf %in, %in_5 : f32
        linalg.yield %11 : f32
      } -> tensor<64x64xf32>
      %extracted_slice_4 = tensor.extract_slice %arg6[%arg4, %arg5] [64, 64] [1, 1] : tensor<512x512xf32> to tensor<64x64xf32>
      %10 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"], library_call = "relu"} ins(%9 : tensor<64x64xf32>) outs(%extracted_slice_4 : tensor<64x64xf32>) {
      ^bb0(%in: f32, %out: f32):
        %11 = arith.maximumf %in, %cst : f32
        linalg.yield %11 : f32
      } -> tensor<64x64xf32>
      scf.forall.in_parallel {
        tensor.parallel_insert_slice %10 into %arg6[%arg4, %arg5] [64, 64] [1, 1] : tensor<64x64xf32> into tensor<512x512xf32>
      }
    }
    bufferization.materialize_in_destination %6 in restrict writable %arg3 : (tensor<512x512xf32>, memref<512x512xf32>) -> ()
    return
  }
}

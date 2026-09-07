// Hand-fixed variant of out/pack_lowered.mlir: eliminates the epilogue
// linalg.copy that transform.structured.lower_unpack always inserts.
//
// lower_unpack unconditionally allocates a fresh tensor.empty() as the
// destination for its synthesized linalg.transpose, then collapse_shapes
// and linalg.copys the result into the unpack's real `dest` operand - even
// when that `dest` is already the function's actual output tensor. Verified
// this is unconditional (independent of what `dest` is wired to) against
// this repo's local LLVM build (llvm-project @ 2b1ebef8, 2025-05-28).
//
// Upstream fixed the equivalent redundant-tensor.empty problem for the
// *vectorized* unpack lowering path (vectorizeAsTensorUnpackOp) in
// llvm/llvm-project#149156 (merged 2025-07-17, after this build) - not
// present here yet. This file is the same fix applied by hand to the
// lower_unpack (non-vectorized-epilogue) path instead: feed the transpose's
// `outs` a tensor.expand_shape of the real destination (%2, i.e. %arg2)
// instead of a scratch tensor.empty, so bufferization writes the transpose
// straight into the caller's output buffer with zero copies.
//
// Diff against out/pack_lowered.mlir is only the last 4 lines before
// `return` (originally):
//   %7 = tensor.empty() : tensor<32x32x32x32xf32>
//   %transposed_2 = linalg.transpose ins(%6 : ...) outs(%7 : ...) permutation = [0, 2, 1, 3]
//   %collapsed = tensor.collapse_shape %transposed_2 [[0, 1], [2, 3]] : ... into tensor<1024x1024xf32>
//   %8 = linalg.copy ins(%collapsed : ...) outs(%2 : ...) -> tensor<1024x1024xf32>
//   bufferization.materialize_in_destination %8 in writable %arg2 : ...
//
// Verified this bufferizes with zero linalg.copy/memref.copy end-to-end:
//   mlir-opt pack_lowered_no_copy.mlir \
//     --one-shot-bufferize="bufferize-function-boundaries=true function-boundary-type-conversion=identity-layout-map" \
//     --cse --canonicalize --buffer-deallocation-pipeline
// -> the epilogue transpose writes directly into `memref.expand_shape %arg2`,
// no memref.copy anywhere in the output.
module {
  func.func @matmul(%arg0: memref<1024x1024xf32>, %arg1: memref<1024x1024xf32>, %arg2: memref<1024x1024xf32>) {
    %c8 = arith.constant 8 : index
    %c1 = arith.constant 1 : index
    %c32 = arith.constant 32 : index
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.000000e+00 : f32
    %0 = bufferization.to_tensor %arg0 restrict : memref<1024x1024xf32> to tensor<1024x1024xf32>
    %1 = bufferization.to_tensor %arg1 restrict : memref<1024x1024xf32> to tensor<1024x1024xf32>
    %2 = bufferization.to_tensor %arg2 restrict writable : memref<1024x1024xf32> to tensor<1024x1024xf32>
    %3 = tensor.empty() : tensor<1024x1024xf32>
    %4 = tensor.empty() : tensor<32x32x32x32xf32>
    %expanded = tensor.expand_shape %0 [[0, 1], [2, 3]] output_shape [32, 32, 32, 32] : tensor<1024x1024xf32> into tensor<32x32x32x32xf32>
    %transposed = linalg.transpose ins(%expanded : tensor<32x32x32x32xf32>) outs(%4 : tensor<32x32x32x32xf32>) permutation = [0, 2, 1, 3]
    %expanded_0 = tensor.expand_shape %1 [[0, 1], [2, 3]] output_shape [32, 32, 32, 32] : tensor<1024x1024xf32> into tensor<32x32x32x32xf32>
    %transposed_1 = linalg.transpose ins(%expanded_0 : tensor<32x32x32x32xf32>) outs(%4 : tensor<32x32x32x32xf32>) permutation = [0, 2, 3, 1]
    %5 = linalg.fill ins(%cst : f32) outs(%4 : tensor<32x32x32x32xf32>) -> tensor<32x32x32x32xf32>
    %6 = scf.for %arg3 = %c0 to %c32 step %c1 iter_args(%arg4 = %5) -> (tensor<32x32x32x32xf32>) {
      %9 = scf.for %arg5 = %c0 to %c32 step %c1 iter_args(%arg6 = %arg4) -> (tensor<32x32x32x32xf32>) {
        %10 = scf.for %arg7 = %c0 to %c32 step %c1 iter_args(%arg8 = %arg6) -> (tensor<32x32x32x32xf32>) {
          %extracted_slice = tensor.extract_slice %transposed[%arg3, %arg7, 0, 0] [1, 1, 32, 32] [1, 1, 1, 1] : tensor<32x32x32x32xf32> to tensor<32x32xf32>
          %extracted_slice_3 = tensor.extract_slice %transposed_1[%arg7, %arg5, 0, 0] [1, 1, 32, 32] [1, 1, 1, 1] : tensor<32x32x32x32xf32> to tensor<32x32xf32>
          %extracted_slice_4 = tensor.extract_slice %arg8[%arg3, %arg5, 0, 0] [1, 1, 32, 32] [1, 1, 1, 1] : tensor<32x32x32x32xf32> to tensor<32x32xf32>
          %11 = scf.for %arg9 = %c0 to %c32 step %c8 iter_args(%arg10 = %extracted_slice_4) -> (tensor<32x32xf32>) {
            %12 = scf.for %arg11 = %c0 to %c32 step %c8 iter_args(%arg12 = %arg10) -> (tensor<32x32xf32>) {
              %13 = scf.for %arg13 = %c0 to %c32 step %c8 iter_args(%arg14 = %arg12) -> (tensor<32x32xf32>) {
                %extracted_slice_5 = tensor.extract_slice %extracted_slice[%arg9, %arg13] [8, 8] [1, 1] : tensor<32x32xf32> to tensor<8x8xf32>
                %extracted_slice_6 = tensor.extract_slice %extracted_slice_3[%arg11, %arg13] [8, 8] [1, 1] : tensor<32x32xf32> to tensor<8x8xf32>
                %extracted_slice_7 = tensor.extract_slice %arg14[%arg9, %arg11] [8, 8] [1, 1] : tensor<32x32xf32> to tensor<8x8xf32>
                %14 = vector.transfer_read %extracted_slice_5[%c0, %c0], %cst {in_bounds = [true, true]} : tensor<8x8xf32>, vector<8x8xf32>
                %15 = vector.transfer_read %extracted_slice_6[%c0, %c0], %cst {in_bounds = [true, true]} : tensor<8x8xf32>, vector<8x8xf32>
                %16 = vector.transfer_read %extracted_slice_7[%c0, %c0], %cst {in_bounds = [true, true]} : tensor<8x8xf32>, vector<8x8xf32>
                %17 = vector.transpose %14, [1, 0] : vector<8x8xf32> to vector<8x8xf32>
                %18 = vector.transpose %15, [1, 0] : vector<8x8xf32> to vector<8x8xf32>
                %19 = vector.extract %17[0] : vector<8xf32> from vector<8x8xf32>
                %20 = vector.extract %18[0] : vector<8xf32> from vector<8x8xf32>
                %21 = vector.outerproduct %19, %20, %16 {kind = #vector.kind<add>} : vector<8xf32>, vector<8xf32>
                %22 = vector.extract %17[1] : vector<8xf32> from vector<8x8xf32>
                %23 = vector.extract %18[1] : vector<8xf32> from vector<8x8xf32>
                %24 = vector.outerproduct %22, %23, %21 {kind = #vector.kind<add>} : vector<8xf32>, vector<8xf32>
                %25 = vector.extract %17[2] : vector<8xf32> from vector<8x8xf32>
                %26 = vector.extract %18[2] : vector<8xf32> from vector<8x8xf32>
                %27 = vector.outerproduct %25, %26, %24 {kind = #vector.kind<add>} : vector<8xf32>, vector<8xf32>
                %28 = vector.extract %17[3] : vector<8xf32> from vector<8x8xf32>
                %29 = vector.extract %18[3] : vector<8xf32> from vector<8x8xf32>
                %30 = vector.outerproduct %28, %29, %27 {kind = #vector.kind<add>} : vector<8xf32>, vector<8xf32>
                %31 = vector.extract %17[4] : vector<8xf32> from vector<8x8xf32>
                %32 = vector.extract %18[4] : vector<8xf32> from vector<8x8xf32>
                %33 = vector.outerproduct %31, %32, %30 {kind = #vector.kind<add>} : vector<8xf32>, vector<8xf32>
                %34 = vector.extract %17[5] : vector<8xf32> from vector<8x8xf32>
                %35 = vector.extract %18[5] : vector<8xf32> from vector<8x8xf32>
                %36 = vector.outerproduct %34, %35, %33 {kind = #vector.kind<add>} : vector<8xf32>, vector<8xf32>
                %37 = vector.extract %17[6] : vector<8xf32> from vector<8x8xf32>
                %38 = vector.extract %18[6] : vector<8xf32> from vector<8x8xf32>
                %39 = vector.outerproduct %37, %38, %36 {kind = #vector.kind<add>} : vector<8xf32>, vector<8xf32>
                %40 = vector.extract %17[7] : vector<8xf32> from vector<8x8xf32>
                %41 = vector.extract %18[7] : vector<8xf32> from vector<8x8xf32>
                %42 = vector.outerproduct %40, %41, %39 {kind = #vector.kind<add>} : vector<8xf32>, vector<8xf32>
                %43 = vector.transfer_write %42, %extracted_slice_7[%c0, %c0] {in_bounds = [true, true]} : vector<8x8xf32>, tensor<8x8xf32>
                %inserted_slice_8 = tensor.insert_slice %43 into %arg14[%arg9, %arg11] [8, 8] [1, 1] : tensor<8x8xf32> into tensor<32x32xf32>
                scf.yield %inserted_slice_8 : tensor<32x32xf32>
              }
              scf.yield %13 : tensor<32x32xf32>
            }
            scf.yield %12 : tensor<32x32xf32>
          }
          %inserted_slice = tensor.insert_slice %11 into %arg8[%arg3, %arg5, 0, 0] [1, 1, 32, 32] [1, 1, 1, 1] : tensor<32x32xf32> into tensor<32x32x32x32xf32>
          scf.yield %inserted_slice : tensor<32x32x32x32xf32>
        }
        scf.yield %10 : tensor<32x32x32x32xf32>
      }
      scf.yield %9 : tensor<32x32x32x32xf32>
    }
    // --- fix starts here: feed the epilogue transpose a view of the real
    // destination instead of a scratch tensor.empty, and drop the trailing
    // linalg.copy that lower_unpack would otherwise insert. ---
    %7 = tensor.expand_shape %2 [[0, 1], [2, 3]] output_shape [32, 32, 32, 32] : tensor<1024x1024xf32> into tensor<32x32x32x32xf32>
    %transposed_2 = linalg.transpose ins(%6 : tensor<32x32x32x32xf32>) outs(%7 : tensor<32x32x32x32xf32>) permutation = [0, 2, 1, 3]
    %collapsed = tensor.collapse_shape %transposed_2 [[0, 1], [2, 3]] : tensor<32x32x32x32xf32> into tensor<1024x1024xf32>
    bufferization.materialize_in_destination %collapsed in writable %arg2 : (tensor<1024x1024xf32>, memref<1024x1024xf32>) -> ()
    return
  }
}

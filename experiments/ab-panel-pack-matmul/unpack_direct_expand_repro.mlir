// Validates the custom-lowering idea in isolation before writing it as a
// real RewritePattern: instead of lowerUnPack's fresh tensor.empty() for the
// transpose's scratch output, expand_shape the *real* destination (%C) up
// to the strip-mined shape and transpose straight into that view. Shapes/
// permutation/reassociations here match linalg.unpack's C-side decomposition
// in out/pack_lowered.mlir exactly (permutation = [0,2,1,3], reassociations
// [[0,1],[2,3]]) - only the outs operand of the transpose differs.
func.func @unpack_direct_expand(%source_memref: memref<32x32x32x32xf32>, %C: memref<1024x1024xf32>) {
  %source = bufferization.to_tensor %source_memref restrict : memref<32x32x32x32xf32> to tensor<32x32x32x32xf32>
  %dest = bufferization.to_tensor %C restrict writable : memref<1024x1024xf32> to tensor<1024x1024xf32>

  %expanded_dest = tensor.expand_shape %dest [[0, 1], [2, 3]] output_shape [32, 32, 32, 32]
      : tensor<1024x1024xf32> into tensor<32x32x32x32xf32>

  %transposed = linalg.transpose ins(%source : tensor<32x32x32x32xf32>)
                outs(%expanded_dest : tensor<32x32x32x32xf32>) permutation = [0, 2, 1, 3]

  %collapsed = tensor.collapse_shape %transposed [[0, 1], [2, 3]]
               : tensor<32x32x32x32xf32> into tensor<1024x1024xf32>

  bufferization.materialize_in_destination %collapsed in restrict writable %C
      : (tensor<1024x1024xf32>, memref<1024x1024xf32>) -> ()
  return
}

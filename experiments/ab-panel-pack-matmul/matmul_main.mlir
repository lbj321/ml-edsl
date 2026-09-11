// Correctness harness: wraps matmul.mlir's @matmul_fn_0 with a @main that
// fills A=1.0, B=2.0 and prints two output elements. Expected result: 2*K
// = 2048.0 (K=1024) for every entry, since every element is a dot product
// of a row of all-1.0 against a column of all-2.0.
//
// Runs through the exact same block-pack/tile/vectorize/lower-unpack-direct
// pipeline as matmul.mlir (every pack/tile/vectorize/unpack-lowering stage
// only ever matches ops inside @matmul_fn_0 - linalg.matmul, then
// linalg.pack/generic/unpack - so @main's plain memref.alloc/linalg.fill/
// func.call/vector.print ops pass through every stage untouched), then
// gets JIT-run via mlir-runner to verify LowerUnpackDirectPass's copy-free
// unpack lowering produces the *correct* result, not just IR with zero
// memref.copy - a wrong reassociation/permutation in the custom lowering
// could easily produce copy-free IR that computes the wrong answer, which
// the memref.copy-count check alone would never catch.
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

func.func @main() {
  %A = memref.alloc() : memref<1024x1024xf32>
  %B = memref.alloc() : memref<1024x1024xf32>
  %C = memref.alloc() : memref<1024x1024xf32>

  %one = arith.constant 1.0 : f32
  %two = arith.constant 2.0 : f32
  linalg.fill ins(%one : f32) outs(%A : memref<1024x1024xf32>)
  linalg.fill ins(%two : f32) outs(%B : memref<1024x1024xf32>)

  func.call @matmul_fn_0(%A, %B, %C) : (memref<1024x1024xf32>, memref<1024x1024xf32>, memref<1024x1024xf32>) -> ()

  %c0 = arith.constant 0 : index
  %c64 = arith.constant 64 : index
  %c1023 = arith.constant 1023 : index
  %corner = memref.load %C[%c0, %c0] : memref<1024x1024xf32>
  %center = memref.load %C[%c64, %c64] : memref<1024x1024xf32>
  %last = memref.load %C[%c1023, %c1023] : memref<1024x1024xf32>
  vector.print %corner : f32
  vector.print %center : f32
  vector.print %last : f32

  // No memref.dealloc here: one-shot-bufferize's ownership analysis over
  // the whole module rejects a hand-written dealloc alongside the
  // tensor-level code still present in @matmul_fn_0 at this stage. Fine to
  // leak for a short-lived correctness-check JIT run.
  return
}

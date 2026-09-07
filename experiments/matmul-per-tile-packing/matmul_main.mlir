// Correctness harness: wraps matmul.mlir's @matmul with a @main that fills
// A=1.0, B=2.0 and prints one output element. Expected result: 2*K = 256.0
// (K=128) for every entry, since every element is a dot product of a
// row of all-1.0 against a column of all-2.0.
//
// Runs through the exact same transform-dialect/bufferize/lower-to-llvm
// pipeline as matmul.mlir (pack.mlir/tile.mlir/vectorize.mlir only ever
// match ops inside @matmul - linalg.matmul, then linalg.generic - so
// @main's plain memref.alloc/linalg.fill/func.call/vector.print ops pass
// through untouched), then gets JIT-run via mlir-runner to verify the
// pack -> tile -> vectorize -> unpack round-trip produces the correct
// result, not just IR that parses.
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

func.func @main() {
  %A = memref.alloc() : memref<1024x1024xf32>
  %B = memref.alloc() : memref<1024x1024xf32>
  %C = memref.alloc() : memref<1024x1024xf32>

  %one = arith.constant 1.0 : f32
  %two = arith.constant 2.0 : f32
  linalg.fill ins(%one : f32) outs(%A : memref<1024x1024xf32>)
  linalg.fill ins(%two : f32) outs(%B : memref<1024x1024xf32>)

  func.call @matmul(%A, %B, %C) : (memref<1024x1024xf32>, memref<1024x1024xf32>, memref<1024x1024xf32>) -> ()

  %c0 = arith.constant 0 : index
  %c64 = arith.constant 64 : index
  %corner = memref.load %C[%c0, %c0] : memref<1024x1024xf32>
  %center = memref.load %C[%c64, %c64] : memref<1024x1024xf32>
  vector.print %corner : f32
  vector.print %center : f32

  // No memref.dealloc here: one-shot-bufferize's ownership analysis over
  // the whole module rejects a hand-written dealloc alongside the
  // tensor-level code still present in @matmul at this stage (fails with
  // "memory free side-effect on MemRef value not supported"). Fine to
  // leak for a short-lived correctness-check JIT run.
  return
}

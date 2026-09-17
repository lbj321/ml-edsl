#!/usr/bin/env bash
# Reproduces the memref.alloca-in-loop stack-overflow bug at N=512 by running
# mlir-edsl-opt through the *exact* pass sequence buildCPUPipeline uses
# (cpp/src/MLIRLowering.cpp), with linalg-tile-matmul-k skipped to match its
# current disabled state.
#
# Dumps intermediate stages around the interesting boundary:
#   - after alloca-scope-cleanup (the memref.alloca_scope from
#     convert-scf-to-openmp gets inlined away HERE — before any memref.alloca
#     exists to justify keeping it)
#   - after convert-vector-to-scf (THIS is what creates the memref.alloca ops
#     — 4 of them land inside the K-reduction scf.for body, with no scope left
#     to bracket them)
#   - final LLVM dialect IR (the allocas are now bare `llvm.alloca` inside the
#     loop body block, not hoisted to the entry block — each of the 64
#     K-iterations at N=512 permanently grows the stack instead of reusing a
#     slot)
#
# Build mlir-edsl-opt first: ./build.sh
set -euo pipefail

OPT="${MLIR_EDSL_OPT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)/build/cpp/tools/mlir-edsl-opt/mlir-edsl-opt}"
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="$DIR/../out"
mkdir -p "$OUT"

PRE_BUFFERIZE=(
  --linalg-epilogue-tile-and-fuse --canonicalize
  --linalg-tile-generic --canonicalize
  --linalg-matmul-to-contract --canonicalize
  --linalg-vectorize --canonicalize
  --vector-cleanup
  --vector-contract-to-outerproduct
  --lower-vector-multi-reduction
  --eliminate-empty-tensors
)

BUFFERIZE=(
  --one-shot-bufferize="bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map"
  --cse --canonicalize
  --ownership-based-buffer-deallocation
  --buffer-deallocation-simplification
  --bufferization-lower-deallocations
)

PARALLELIZE=(
  --scf-forall-to-parallel
  --convert-scf-to-openmp
)

"$OPT" "$DIR/input_512.mlir" \
  "${PRE_BUFFERIZE[@]}" "${BUFFERIZE[@]}" "${PARALLELIZE[@]}" \
  --alloca-scope-cleanup \
  -o "$OUT/1_post_alloca_scope_cleanup_512.mlir"
echo "wrote $OUT/1_post_alloca_scope_cleanup_512.mlir (no memref.alloca yet — scope already stripped)"

"$OPT" "$OUT/1_post_alloca_scope_cleanup_512.mlir" \
  --convert-linalg-to-loops \
  --convert-vector-to-scf \
  -o "$OUT/2_post_convert_vector_to_scf_512.mlir"
echo "wrote $OUT/2_post_convert_vector_to_scf_512.mlir (memref.alloca now inside K-loop, unscoped)"

"$OPT" "$OUT/2_post_convert_vector_to_scf_512.mlir" \
  --canonicalize \
  --convert-vector-to-llvm="enable-x86vector" \
  --convert-ub-to-llvm \
  --convert-scf-to-cf \
  --expand-strided-metadata \
  --lower-affine \
  --convert-arith-to-llvm \
  --finalize-memref-to-llvm \
  --convert-cf-to-llvm \
  --convert-func-to-llvm \
  --convert-openmp-to-llvm \
  --reconcile-unrealized-casts \
  -o "$OUT/3_final_llvm_dialect_512.mlir"
echo "wrote $OUT/3_final_llvm_dialect_512.mlir (llvm.alloca sitting in the loop body block, not the entry block)"

echo
echo "Compare against a working size (e.g. sed 's/512/384/g' input_512.mlir > input_384.mlir and rerun) to see the same allocas present but with a shorter-running loop."

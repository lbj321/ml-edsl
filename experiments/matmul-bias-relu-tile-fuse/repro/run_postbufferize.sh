#!/usr/bin/env bash
# Runs mlir-edsl-opt on input_512.mlir (pass-2 state: after outer 64x64
# tile+fuse, before inner tiling — linalg-tile-matmul-k is currently disabled
# in buildCPUPipeline, see MLIRLowering.cpp) through one-shot-bufferize, and
# dumps the result for inspection.
#
# Build mlir-edsl-opt first: ./build.sh (binary lands at
# build/cpp/tools/mlir-edsl-opt/mlir-edsl-opt)
set -euo pipefail

OPT="${MLIR_EDSL_OPT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)/build/cpp/tools/mlir-edsl-opt/mlir-edsl-opt}"
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="$DIR/../out"
mkdir -p "$OUT"

"$OPT" "$DIR/input_512.mlir" \
  --linalg-epilogue-tile-and-fuse --canonicalize \
  --linalg-tile-generic --canonicalize \
  --linalg-matmul-to-contract --canonicalize \
  --linalg-vectorize --canonicalize \
  --vector-cleanup \
  --vector-contract-to-outerproduct \
  --lower-vector-multi-reduction \
  --eliminate-empty-tensors \
  --one-shot-bufferize="bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map" \
  --cse --canonicalize \
  -o "$OUT/postbufferize_512.mlir"

echo "wrote $OUT/postbufferize_512.mlir"

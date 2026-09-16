#!/usr/bin/env bash
# Applies the inner tile+fuse transform script to the pass-2 IR and dumps
# each stage to ../out/ for inspection.
#
# mlir-opt must be on PATH (or point MLIR_OPT at the LLVM build's bin dir),
# e.g.: export MLIR_OPT=$LLVM_BUILD_DIR/bin/mlir-opt
set -euo pipefail

MLIR_OPT="${MLIR_OPT:-mlir-opt}"
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="$DIR/../out"
mkdir -p "$OUT"

"$MLIR_OPT" "$DIR/input.mlir" \
  -transform-preload-library="transform-library-paths=$DIR/transform.mlir" \
  -transform-interpreter \
  -canonicalize -cse \
  -o "$OUT/fused.mlir"

echo "wrote $OUT/fused.mlir"

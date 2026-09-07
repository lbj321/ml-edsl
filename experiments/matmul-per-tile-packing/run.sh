#!/usr/bin/env bash
# Runs matmul.mlir through the full pack -> tile -> vectorize -> lower pipeline,
# writing intermediate IR to out/ at each stage. Mirrors the stage-by-stage
# structure used to isolate the N>=192 O0 stack-overflow bug (see pack.mlir/
# tile.mlir/vectorize.mlir/pack_lowering.mlir for what each stage does and why).
#
# Usage:
#   ./run.sh [input.mlir]
#
#   input.mlir   Input file. Default: matmul.mlir (checked into this dir).
#                Block factors (32,32,32) are currently hardcoded in
#                pack.mlir, not parameterized here.
#
# Writes out/{packed,tiled,vectorized,pack_lowered,bufferized,openmp,llvm}.mlir.
#
# Requires MLIR_OPT env var pointing at an LLVM/MLIR build's mlir-opt, or
# edit the default below.

set -euo pipefail

MLIR_OPT="${MLIR_OPT:-$HOME/dev/llvm-project/build/bin/mlir-opt}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INPUT="${1:-$SCRIPT_DIR/matmul.mlir}"
OUT_DIR="$SCRIPT_DIR/out"

if [[ ! -e "$MLIR_OPT" ]]; then
  echo "error: not found: $MLIR_OPT" >&2
  echo "  set MLIR_OPT to override" >&2
  exit 1
fi

mkdir -p "$OUT_DIR"

echo "[1/7] pack (transform.structured.pack, 32x32x32)"
"$MLIR_OPT" "$INPUT" \
  --transform-preload-library="transform-library-paths=$SCRIPT_DIR/pack.mlir" \
  --transform-interpreter \
  -o "$OUT_DIR/packed.mlir"

echo "[2/7] tile (block-grid -> 1, then 32x32x32 -> 8x8x8)"
"$MLIR_OPT" "$OUT_DIR/packed.mlir" \
  --transform-preload-library="transform-library-paths=$SCRIPT_DIR/tile.mlir" \
  --transform-interpreter \
  -o "$OUT_DIR/tiled.mlir"

echo "[3/7] vectorize (-> vector.contract -> vector.outerproduct, pre-bufferize)"
"$MLIR_OPT" "$OUT_DIR/tiled.mlir" \
  --transform-preload-library="transform-library-paths=$SCRIPT_DIR/vectorize.mlir" \
  --transform-interpreter \
  -o "$OUT_DIR/vectorized.mlir"

echo "[4/7] lower pack/unpack + fold no-op pad"
"$MLIR_OPT" "$OUT_DIR/vectorized.mlir" \
  --transform-preload-library="transform-library-paths=$SCRIPT_DIR/pack_lowering.mlir" \
  --transform-interpreter \
  --canonicalize \
  --eliminate-empty-tensors \
  -o "$OUT_DIR/pack_lowered.mlir"

echo "[5/7] bufferize (identity-layout-map, matching addCPUPasses)"
"$MLIR_OPT" "$OUT_DIR/pack_lowered.mlir" \
  --one-shot-bufferize="bufferize-function-boundaries=true function-boundary-type-conversion=identity-layout-map" \
  --cse \
  --canonicalize \
  --buffer-deallocation-pipeline \
  -o "$OUT_DIR/bufferized.mlir"

echo "[6/7] scf.forall -> scf.parallel -> omp.parallel"
# tile.mlir's outer M/N block-grid tiling produced scf.forall (see tile.mlir
# for why K block-grid stays a serial scf.for instead). --canonicalize here
# inlines the alloca_scope that convert-scf-to-openmp always wraps the body
# in (AllocaScopeOp's own AllocaScopeInliner canonicalization pattern fires
# because the scope is still empty at this point - nothing has introduced a
# memref.alloca yet) - matching AllocaScopeCleanupPass's job in
# cpp/src/MLIRLowering.cpp, but the empty scope means stock canonicalize does
# it too, no custom pass needed here.
"$MLIR_OPT" "$OUT_DIR/bufferized.mlir" \
  --scf-forall-to-parallel \
  --convert-scf-to-openmp \
  --canonicalize \
  -o "$OUT_DIR/openmp.mlir"

echo "[7/7] lower to LLVM dialect"
"$MLIR_OPT" "$OUT_DIR/openmp.mlir" \
  --convert-linalg-to-loops \
  --convert-vector-to-scf \
  --buffer-loop-hoisting \
  --canonicalize \
  --convert-scf-to-cf \
  --expand-strided-metadata \
  --lower-affine \
  --convert-vector-to-llvm \
  --convert-ub-to-llvm \
  --convert-arith-to-llvm \
  --finalize-memref-to-llvm \
  --convert-cf-to-llvm \
  --convert-func-to-llvm \
  --convert-openmp-to-llvm \
  --reconcile-unrealized-casts \
  --canonicalize --cse \
  -o "$OUT_DIR/llvm.mlir"

echo
echo "done. IR at each stage: $OUT_DIR/{packed,tiled,vectorized,pack_lowered,bufferized,openmp,llvm}.mlir"
echo "run: mlir-runner $OUT_DIR/llvm.mlir -e main -entry-point-result=void --O3 \\"
echo "       --shared-libs=<llvm-build>/lib/libmlir_runner_utils.so,<llvm-build>/lib/libmlir_c_runner_utils.so"

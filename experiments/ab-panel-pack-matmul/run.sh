#!/usr/bin/env bash
# Runs matmul.mlir through the -linalg-block-pack-matmul pipeline, writing
# intermediate IR to out/ at each stage. Companion to
# ../matmul-per-tile-packing/run.sh: that one hand-rolls pack/tile/vectorize/
# lower via transform-dialect libraries with independent M/N/K pack control;
# this one exercises the upstream -linalg-block-pack-matmul pass instead,
# which packs A, B, *and* C in one shot off a single block-factors triple (no
# per-operand opt-out) but already picks BLIS-style orientation (B's
# outer/inner dims transposed relative to A's) without any flags set - see
# out/packed.mlir once stage 1 has run.
#
# Stages so far: pack -> tile (M/N forall only) -> lower A/B's linalg.pack
# (stock) -> lower C's linalg.unpack via our own LowerUnpackDirectPass
# (standalone-opt, built from CMakeLists.txt in this dir) -> bufferize.
# LowerUnpackDirectPass avoids the scratch tensor.empty stock lowerUnPack
# allocates for its internal transpose (which sits behind a
# collapse_shape/expand_shape -eliminate-empty-tensors can never see through
# - see LowerUnpackDirectPass.cpp's file comment for the full story), so C's
# unpack should bufferize with zero memref.copy into %arg2. The final check
# below greps for exactly that.
#
# More stages (K block-grid tiling, 8x8x8 tiling, vectorization) get added
# here as they're worked out, mirroring how matmul-per-tile-packing/run.sh
# grew one stage at a time.
#
# Usage:
#   ./run.sh [input.mlir]
#
#   input.mlir   Input file. Default: matmul.mlir (checked into this dir).
#                Block factors (32,32,32) are currently hardcoded below.
#
# Writes out/{packed,tiled_mn,ablowered,unpack_direct,bufferized}.mlir.
#
# Requires MLIR_OPT env var pointing at an LLVM/MLIR build's mlir-opt (or
# edit the default below), and build-standalone/standalone-opt built first:
#   cmake -S . -B build-standalone -G Ninja && ninja -C build-standalone

set -euo pipefail

MLIR_OPT="${MLIR_OPT:-$HOME/dev/llvm-project/build/bin/mlir-opt}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INPUT="${1:-$SCRIPT_DIR/matmul.mlir}"
OUT_DIR="$SCRIPT_DIR/out"
STANDALONE_OPT="$SCRIPT_DIR/build-standalone/standalone-opt"

if [[ ! -e "$MLIR_OPT" ]]; then
  echo "error: not found: $MLIR_OPT" >&2
  echo "  set MLIR_OPT to override" >&2
  exit 1
fi
if [[ ! -e "$STANDALONE_OPT" ]]; then
  echo "error: not found: $STANDALONE_OPT" >&2
  echo "  build it first: cmake -S . -B build-standalone -G Ninja && ninja -C build-standalone" >&2
  exit 1
fi

mkdir -p "$OUT_DIR"

echo "[1/5] block-pack (linalg-block-pack-matmul, 32x32x32, default orientation)"
"$MLIR_OPT" "$INPUT" \
  --linalg-block-pack-matmul="block-factors=32,32,32" \
  -o "$OUT_DIR/packed.mlir"

echo "[2/5] tile (outer M/N block-grid -> scf.forall only, so far)"
"$MLIR_OPT" "$OUT_DIR/packed.mlir" \
  --transform-preload-library="transform-library-paths=$SCRIPT_DIR/tile_mn_forall.mlir" \
  --transform-interpreter \
  -o "$OUT_DIR/tiled_mn.mlir"

echo "[3/5] lower A/B's linalg.pack only (stock lower_pack), leave linalg.unpack for stage 4"
"$MLIR_OPT" "$OUT_DIR/tiled_mn.mlir" \
  --transform-preload-library="transform-library-paths=$SCRIPT_DIR/pack_lowering_ab_only.mlir" \
  --transform-interpreter --canonicalize \
  -o "$OUT_DIR/ablowered.mlir"

echo "[4/5] lower C's linalg.unpack directly into its destination (LowerUnpackDirectPass)"
"$STANDALONE_OPT" "$OUT_DIR/ablowered.mlir" \
  --linalg-lower-unpack-direct \
  -o "$OUT_DIR/unpack_direct.mlir"

echo "[5/5] bufferize (identity-layout-map, matching addCPUPasses) + canonicalize/cse"
# Two canonicalize passes: the first alone leaves a self-copy
# (memref.copy %x, %x) inside the forall body from tile_using_forall's own
# bufferization of tensor.parallel_insert_slice - confirmed by hand earlier
# that a second pass folds it away; this isn't related to
# LowerUnpackDirectPass's own fix.
"$MLIR_OPT" "$OUT_DIR/unpack_direct.mlir" \
  --one-shot-bufferize="bufferize-function-boundaries=true function-boundary-type-conversion=identity-layout-map" \
  --canonicalize --cse --canonicalize \
  -o "$OUT_DIR/bufferized.mlir"

echo
echo "done. IR at each stage: $OUT_DIR/{packed,tiled_mn,ablowered,unpack_direct,bufferized}.mlir"
echo
COPY_COUNT=$(grep -c "memref.copy" "$OUT_DIR/bufferized.mlir" || true)
echo "memref.copy count in bufferized.mlir: $COPY_COUNT (expect 0 - into %arg2 should be a bare transpose, no copy)"

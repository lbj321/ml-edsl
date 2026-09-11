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
# Full pipeline: pack -> tile M/N forall -> tile K block-grid (scf.for,
# nested in the forall) -> tile inner 32x32x32 -> 8x8x8 (three more nested
# scf.for) -> vectorize (vector.contract -> vector.outerproduct) -> lower
# A/B's linalg.pack (stock) -> lower C's linalg.unpack via our own
# LowerUnpackDirectPass (standalone-opt, built from CMakeLists.txt in this
# dir) -> bufferize. LowerUnpackDirectPass avoids the scratch tensor.empty
# stock lowerUnPack allocates for its internal transpose (which sits behind
# a collapse_shape/expand_shape -eliminate-empty-tensors can never see
# through - see LowerUnpackDirectPass.cpp's file comment for the full
# story), so C's unpack should bufferize with zero memref.copy into %arg2.
# The final check below greps for exactly that - confirmed holding through
# the full pipeline (forall + K for-loop + 8x8x8 for-loops + vectorization),
# matching matmul-per-tile-packing/run.sh's stage depth.
#
# Usage:
#   ./run.sh [input.mlir]
#
#   input.mlir   Input file. Default: matmul.mlir (checked into this dir).
#                Block factors (32,32,32) are currently hardcoded below.
#
# Writes out/{packed,tiled_mn,tiled_mn_k,tiled_mn_k_888,vectorized,ablowered,unpack_direct,bufferized}.mlir.
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

echo "[1/8] block-pack (linalg-block-pack-matmul, 32x32x32, default orientation)"
"$MLIR_OPT" "$INPUT" \
  --linalg-block-pack-matmul="block-factors=32,32,32" \
  -o "$OUT_DIR/packed.mlir"

echo "[2/8] tile outer M/N block-grid -> scf.forall"
"$MLIR_OPT" "$OUT_DIR/packed.mlir" \
  --transform-preload-library="transform-library-paths=$SCRIPT_DIR/tile_mn_forall.mlir" \
  --transform-interpreter \
  -o "$OUT_DIR/tiled_mn.mlir"

echo "[3/8] tile K block-grid -> scf.for (nested inside the forall)"
"$MLIR_OPT" "$OUT_DIR/tiled_mn.mlir" \
  --transform-preload-library="transform-library-paths=$SCRIPT_DIR/tile_k_forloop.mlir" \
  --transform-interpreter \
  -o "$OUT_DIR/tiled_mn_k.mlir"

echo "[4/8] tile inner compute 32x32x32 -> 8x8x8 (vectorization granularity)"
"$MLIR_OPT" "$OUT_DIR/tiled_mn_k.mlir" \
  --transform-preload-library="transform-library-paths=$SCRIPT_DIR/tile_inner_8x8x8.mlir" \
  --transform-interpreter \
  -o "$OUT_DIR/tiled_mn_k_888.mlir"

echo "[5/8] vectorize the tiled 8x8x8 compute generic -> vector.contract -> vector.outerproduct"
"$MLIR_OPT" "$OUT_DIR/tiled_mn_k_888.mlir" \
  --transform-preload-library="transform-library-paths=$SCRIPT_DIR/vectorize.mlir" \
  --transform-interpreter \
  -o "$OUT_DIR/vectorized.mlir"

echo "[6/8] lower A/B's linalg.pack only (stock lower_pack), leave linalg.unpack for stage 7"
"$MLIR_OPT" "$OUT_DIR/vectorized.mlir" \
  --transform-preload-library="transform-library-paths=$SCRIPT_DIR/pack_lowering_ab_only.mlir" \
  --transform-interpreter --canonicalize \
  -o "$OUT_DIR/ablowered.mlir"

echo "[7/8] lower C's linalg.unpack directly into its destination (LowerUnpackDirectPass)"
"$STANDALONE_OPT" "$OUT_DIR/ablowered.mlir" \
  --linalg-lower-unpack-direct \
  -o "$OUT_DIR/unpack_direct.mlir"

echo "[8/8] bufferize (identity-layout-map, matching addCPUPasses) + canonicalize/cse"
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
echo "done. IR at each stage: $OUT_DIR/{packed,tiled_mn,tiled_mn_k,tiled_mn_k_888,vectorized,ablowered,unpack_direct,bufferized}.mlir"
echo
COPY_COUNT=$(grep -c "memref.copy" "$OUT_DIR/bufferized.mlir" || true)
echo "memref.copy count in bufferized.mlir: $COPY_COUNT (expect 0)"
if [[ "$COPY_COUNT" -ne 0 ]]; then
  echo "FAIL: expected 0 memref.copy into the destination, found $COPY_COUNT" >&2
  exit 1
fi

# Correctness check: the copy-count assertion above only proves the IR
# *shape* is copy-free - it says nothing about whether
# LowerUnpackDirectPass's reassociation/permutation handling is actually
# right. Rerun the exact same 8 stages on matmul_main.mlir (matmul_fn_0
# plus a @main that fills A=1.0, B=2.0 and prints 3 output elements - every
# pack/tile/vectorize/unpack-lowering stage only ever matches ops inside
# @matmul_fn_0, so @main passes through untouched), then finish lowering to
# LLVM dialect (--scf-forall-to-for instead of the real pipeline's
# forall->parallel->omp path, to keep this a simple single-threaded
# correctness run with no OpenMP runtime dependency) and JIT-run it.
# Expected: 2*K = 2048.0 at every position, since every output element is
# a dot product of a row of all-1.0 against a column of all-2.0.
MLIR_RUNNER="$(dirname "$MLIR_OPT")/mlir-runner"
LLVM_LIB_DIR="$(dirname "$MLIR_OPT")/../lib"
if [[ ! -e "$MLIR_RUNNER" ]]; then
  echo "warning: mlir-runner not found next to mlir-opt, skipping correctness check" >&2
else
  echo
  echo "=== correctness check (matmul_main.mlir, expect 2048 x3) ==="
  "$MLIR_OPT" "$SCRIPT_DIR/matmul_main.mlir" \
    --linalg-block-pack-matmul="block-factors=32,32,32" \
    -o "$OUT_DIR/main_packed.mlir"
  "$MLIR_OPT" "$OUT_DIR/main_packed.mlir" \
    --transform-preload-library="transform-library-paths=$SCRIPT_DIR/tile_mn_forall.mlir" \
    --transform-interpreter \
    -o "$OUT_DIR/main_tiled_mn.mlir"
  "$MLIR_OPT" "$OUT_DIR/main_tiled_mn.mlir" \
    --transform-preload-library="transform-library-paths=$SCRIPT_DIR/tile_k_forloop.mlir" \
    --transform-interpreter \
    -o "$OUT_DIR/main_tiled_mn_k.mlir"
  "$MLIR_OPT" "$OUT_DIR/main_tiled_mn_k.mlir" \
    --transform-preload-library="transform-library-paths=$SCRIPT_DIR/tile_inner_8x8x8.mlir" \
    --transform-interpreter \
    -o "$OUT_DIR/main_tiled_888.mlir"
  "$MLIR_OPT" "$OUT_DIR/main_tiled_888.mlir" \
    --transform-preload-library="transform-library-paths=$SCRIPT_DIR/vectorize.mlir" \
    --transform-interpreter \
    -o "$OUT_DIR/main_vectorized.mlir"
  "$MLIR_OPT" "$OUT_DIR/main_vectorized.mlir" \
    --transform-preload-library="transform-library-paths=$SCRIPT_DIR/pack_lowering_ab_only.mlir" \
    --transform-interpreter --canonicalize \
    -o "$OUT_DIR/main_ablowered.mlir"
  "$STANDALONE_OPT" "$OUT_DIR/main_ablowered.mlir" \
    --linalg-lower-unpack-direct \
    -o "$OUT_DIR/main_unpack_direct.mlir"
  "$MLIR_OPT" "$OUT_DIR/main_unpack_direct.mlir" \
    --one-shot-bufferize="bufferize-function-boundaries=true function-boundary-type-conversion=identity-layout-map" \
    --canonicalize --cse --canonicalize \
    -o "$OUT_DIR/main_bufferized.mlir"
  "$MLIR_OPT" "$OUT_DIR/main_bufferized.mlir" \
    --scf-forall-to-for \
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
    --reconcile-unrealized-casts \
    --canonicalize --cse \
    -o "$OUT_DIR/main_llvm.mlir"

  RESULT=$("$MLIR_RUNNER" "$OUT_DIR/main_llvm.mlir" \
    -e main -entry-point-result=void --O3 \
    --shared-libs="$LLVM_LIB_DIR/libmlir_runner_utils.so,$LLVM_LIB_DIR/libmlir_c_runner_utils.so")
  echo "$RESULT"
  if [[ "$(echo "$RESULT" | sort -u)" != "2048" ]]; then
    echo "FAIL: expected 2048 at every printed position, got:" >&2
    echo "$RESULT" >&2
    exit 1
  fi
  echo "correctness check passed"
fi

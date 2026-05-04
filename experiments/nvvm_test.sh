#!/usr/bin/env bash
# Isolated single-pipeline test for GPU NVVM lowering.
#
# Runs: transform → bufferize (infer-layout-map) → promote-to-stack →
#       linalg-to-loops → gpu-outlining → NVVM lowering
# and checks whether valid LLVM dialect (llvm.func) is emitted.
#
# Usage:
#   ./experiments/nvvm_test.sh
#   MLIR_OPT=/path/to/mlir-opt ./experiments/nvvm_test.sh
#   SAVE_OUTLINED=1 ./experiments/nvvm_test.sh   # keep gpu_outlined.mlir on disk

set -euo pipefail

MLIR_OPT=${MLIR_OPT:-mlir-opt}
DIR=$(cd "$(dirname "$0")" && pwd)
INPUT="$DIR/outparam_test.mlir"

if ! command -v "$MLIR_OPT" &>/dev/null; then
    echo "error: mlir-opt not found. Set MLIR_OPT=/path/to/mlir-opt" >&2
    exit 1
fi

green() { printf '\033[32m%s\033[0m\n' "$*"; }
red()   { printf '\033[31m%s\033[0m\n' "$*"; }
sep()   { printf '\n%s\n\n' "$(printf '─%.0s' {1..72})"; }

# ── Stage 1: bufferize + GPU outlining ───────────────────────────────────────
OUTLINED=$(mktemp /tmp/gpu_outlined_XXXXXX.mlir)

echo "Stage 1: transform → bufferize (infer-layout-map) → linalg-to-loops → gpu-outlining"

"$MLIR_OPT" \
    --transform-interpreter \
    --canonicalize \
    --eliminate-empty-tensors \
    "--one-shot-bufferize=bufferize-function-boundaries=true function-boundary-type-conversion=infer-layout-map" \
    --buffer-results-to-out-params \
    --cse \
    --canonicalize \
    --promote-buffers-to-stack \
    --canonicalize \
    --scf-forall-to-parallel \
    --convert-linalg-to-parallel-loops \
    --gpu-map-parallel-loops \
    --convert-parallel-loops-to-gpu \
    --gpu-kernel-outlining \
    --canonicalize \
    "$INPUT" > "$OUTLINED" 2>&1 || {
        red "[ERROR] Stage 1 failed"
        cat "$OUTLINED"
        rm -f "$OUTLINED"
        exit 1
    }

green "[OK]   Stage 1 passed — outlined IR written"

# Confirm whether memref.copy survived bufferization.
if grep -q "memref\.copy" "$OUTLINED"; then
    red "[WARN] memref.copy present in outlined IR — will become @memrefCopy in PTX"
    grep "memref\.copy" "$OUTLINED"
else
    green "[OK]   No memref.copy — in-place bufferization succeeded"
fi

sep
echo "gpu.module before NVVM:"
grep -A 50 "gpu\.module\b" "$OUTLINED" | head -50

# ── Stage 2: NVVM lowering ────────────────────────────────────────────────────
sep
echo "Stage 2: NVVM lowering"

# convert-gpu-to-nvvm is nested inside gpu.module (mirrors pm.nest<GPUModuleOp>).
# The remaining passes run at builtin.module level and recurse into gpu.module.
NVVM_PIPELINE="builtin.module(
  gpu.module(convert-gpu-to-nvvm),
  convert-scf-to-cf,
  expand-strided-metadata,
  lower-affine,
  convert-arith-to-llvm,
  finalize-memref-to-llvm,
  convert-cf-to-llvm,
  reconcile-unrealized-casts
)"

ir=""
exit_code=0
ir=$("$MLIR_OPT" --pass-pipeline="$NVVM_PIPELINE" "$OUTLINED" 2>&1) || exit_code=$?

if [[ "${SAVE_OUTLINED:-0}" == "1" ]]; then
    cp "$OUTLINED" "$DIR/gpu_outlined.mlir"
    echo "Outlined IR saved to experiments/gpu_outlined.mlir"
fi
rm -f "$OUTLINED"

if [[ $exit_code -ne 0 ]]; then
    red "[ERROR] Stage 2 (NVVM lowering) failed (exit $exit_code)"
    echo "$ir"
    exit 1
fi

sep
echo "$ir"
sep

if echo "$ir" | grep -qE "llvm\.func\b"; then
    green "[OK]   llvm.func found — valid LLVM dialect emitted"
    echo ""
    echo "Signatures:"
    echo "$ir" | grep "llvm\.func\b" | head -10
else
    red "[FAIL] No llvm.func — NVVM lowering incomplete"
    exit 1
fi

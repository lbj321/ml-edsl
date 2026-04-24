#!/usr/bin/env bash
# Demonstrates tile-and-fuse of matmul + bias_add + relu using the transform dialect.
#
# Three schedules are shown:
#
#   BASELINE  — 64×64, one-level fusion, 16×16 tiles
#   CPU       — 128×128, two-level hierarchical: outer 64×64 + inner 8×8×8 matmul
#               mirrors LinalgMatmulParallelTilingPass + LinalgMatmulTilingPass
#   GPU       — 128×128, one-level: 32×32 tiles, K untiled
#               mirrors LinalgGPUMatmulTilingPass
#
# Pass ordering (all three schedules):
#   --linalg-fuse-elementwise-ops   pre-merge bias_add + relu → 1 linalg.generic
#   --transform-interpreter          tile merged generic, fuse fill+matmul upward
#   --cse --canonicalize             clean up dead ops
#
# Usage:
#   ./fuse_matmul_bias_relu.sh
#   MLIR_OPT=/path/to/mlir-opt ./fuse_matmul_bias_relu.sh

set -euo pipefail

MLIR_OPT=${MLIR_OPT:-mlir-opt}
# Derive FileCheck from the same bin/ directory as mlir-opt, fall back to PATH.
FILECHECK=${FILECHECK:-"$(dirname "$(command -v "$MLIR_OPT" 2>/dev/null || echo mlir-opt)")/FileCheck"}
if ! command -v "$FILECHECK" &>/dev/null; then
    FILECHECK=${FILECHECK:-FileCheck}
fi

DIR=$(cd "$(dirname "$0")" && pwd)

if ! command -v "$MLIR_OPT" &>/dev/null; then
    echo "error: mlir-opt not found in PATH. Set MLIR_OPT=/path/to/mlir-opt" >&2
    exit 1
fi

COMMON_OPTS=(
    --linalg-fuse-elementwise-ops   # pre-merge bias_add + relu → 1 generic
    --transform-interpreter          # tile + fuse producers
    --cse --canonicalize             # clean up dead ops
)

# ── helpers ──────────────────────────────────────────────────────────────────
sep() { printf '\n%s\n\n' "$(printf '─%.0s' {1..72})"; }

run_and_check() {
    local label="$1"; shift
    local input="$1"; shift
    # remaining args are mlir-opt options

    sep
    echo "$label"
    sep
    local ir
    ir=$("$MLIR_OPT" "$@" "$input" 2>&1)
    echo "$ir"

    if command -v "$FILECHECK" &>/dev/null; then
        sep
        if echo "$ir" | "$FILECHECK" "$input" 2>&1; then
            echo "[OK] FileCheck passed"
        else
            echo "[FAIL] FileCheck failed" >&2
            exit 1
        fi
    else
        echo "(FileCheck not found — skipping checks)"
    fi
}

# ── step 1: show input IR ─────────────────────────────────────────────────────
echo "INPUT IR  (matmul_bias_relu.mlir — before fusion)"
sep
# Strip transform module so we only show the compute IR.
awk '/^module attributes \{transform\.with_named_sequence\}/{exit} {print}' \
    "$DIR/matmul_bias_relu.mlir"

# ── step 2: baseline (64×64, 16×16 tiles) ───────────────────────────────────
run_and_check \
    "BASELINE — 64×64 matmul, tile relu [16×16], fuse bias_add + matmul + fill" \
    "$DIR/matmul_bias_relu.mlir" \
    "${COMMON_OPTS[@]}"

# ── step 3: CPU two-level (128×128, outer 64×64 + inner 8×8×8) ──────────────
run_and_check \
    "CPU — 128×128 matmul, outer 64×64 tiles (fused), inner 8×8×8 matmul tiles" \
    "$DIR/matmul_bias_relu_cpu.mlir" \
    "${COMMON_OPTS[@]}"

# ── step 4: GPU single-level (128×128, 32×32 tiles, K untiled) ──────────────
run_and_check \
    "GPU — 128×128 matmul, 32×32 tiles (K untiled), fuse bias_add + matmul + fill" \
    "$DIR/matmul_bias_relu_gpu.mlir" \
    "${COMMON_OPTS[@]}"

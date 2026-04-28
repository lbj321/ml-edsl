#!/usr/bin/env bash
# Verify the GPU tile-and-fuse lowering pipeline using stock mlir-opt.
#
# Diagnoses why epilogue fusion (tile relu → fuse fill+matmul+bias_add into
# one scf.forall) fails to emit a GPU kernel in the C++ pipeline.
#
# Two variants are compared:
#   A — convert-linalg-to-parallel-loops  (current C++ pipeline)
#   B — convert-linalg-to-loops           (sequential inner loops, proposed fix)
#
# Note: createLinalgGPUMatmulTilingPass is a custom pass not in stock mlir-opt.
# For the 128×128 dense_layer case the matmul inside the forall is already a
# 32×32 tile, so the GPU tiling pass produces a 1×1 forall that canonicalizes
# away — omitting it here is equivalent.
#
# Usage:
#   ./experiments/tile_fuse_gpu_pipeline.sh
#   MLIR_OPT=/path/to/mlir-opt ./experiments/tile_fuse_gpu_pipeline.sh

set -euo pipefail

MLIR_OPT=${MLIR_OPT:-mlir-opt}
DIR=$(cd "$(dirname "$0")" && pwd)
INPUT="$DIR/tile_fuse_gpu_strategy.mlir"
INPUT_2LEVEL="$DIR/tile_fuse_gpu_2level.mlir"

if ! command -v "$MLIR_OPT" &>/dev/null; then
    echo "error: mlir-opt not found in PATH. Set MLIR_OPT=/path/to/mlir-opt" >&2
    exit 1
fi

sep()   { printf '\n%s\n\n' "$(printf '─%.0s' {1..72})"; }
green() { printf '\033[32m%s\033[0m\n' "$*"; }
red()   { printf '\033[31m%s\033[0m\n' "$*"; }

# Mirrors addGPUPreOutliningPasses up to (and including) scf-forall-to-parallel.
# The linalg→loops pass and GPU mapping passes are added by each variant.
COMMON_PASSES=(
    --transform-interpreter
    --canonicalize
    --linalg-fuse-elementwise-ops
    # one-shot-bufferize: mirrors addBufferizationPasses(withOutParams=false)
    "--one-shot-bufferize=bufferize-function-boundaries=true function-boundary-type-conversion=identity-layout-map"
    --ownership-based-buffer-deallocation
    --canonicalize
    --buffer-deallocation-simplification
    --bufferization-lower-deallocations
    --canonicalize
    # Convert the tiled scf.forall (from transform strategy) to scf.parallel.
    --scf-forall-to-parallel
)

GPU_PASSES=(
    --gpu-map-parallel-loops
    --convert-parallel-loops-to-gpu
    --gpu-kernel-outlining
    --canonicalize
)

# Run mlir-opt with the given passes, print IR, report whether a GPU kernel appeared.
run_variant() {
    local label="$1"
    local linalg_pass="$2"   # --convert-linalg-to-parallel-loops OR --convert-linalg-to-loops

    sep
    echo "VARIANT: $label"
    sep

    local ir exit_code=0
    ir=$("$MLIR_OPT" \
            "${COMMON_PASSES[@]}" \
            "$linalg_pass" \
            "${GPU_PASSES[@]}" \
            "$INPUT" 2>&1) || exit_code=$?

    if [[ $exit_code -ne 0 ]]; then
        red "[ERROR] mlir-opt returned exit code $exit_code"
        echo "$ir"
        return
    fi

    echo "$ir"
    sep

    if echo "$ir" | grep -qE "gpu\.(func|launch_func)\b"; then
        green "[OK]   gpu.func / gpu.launch_func found — kernel emitted"
    else
        red   "[FAIL] No gpu.func in output — no kernel emitted"
    fi
}

# ── Stage 1: show scf.forall structure after transform + bufferize ────────────
sep
echo "STAGE 1 — scf.forall after transform strategy + bufferize"
echo "          (expected: one scf.forall [4,4] containing fill+matmul+bias+relu)"
sep
"$MLIR_OPT" \
    "${COMMON_PASSES[@]}" \
    "$INPUT" 2>&1

# ── Stage 2: show parallel loop nesting (without GPU mapping) ─────────────────
sep
echo "STAGE 2 — parallel loop structure after convert-linalg-to-parallel-loops"
echo "          (shows whether inner loops are nested or sibling)"
sep
"$MLIR_OPT" \
    "${COMMON_PASSES[@]}" \
    --convert-linalg-to-parallel-loops \
    "$INPUT" 2>&1 | grep -E "^\s*(func\.func|scf\.(parallel|for|forall))" || \
    echo "(grep matched nothing)"

# ── Variant A: current C++ pipeline ──────────────────────────────────────────
run_variant \
    "convert-linalg-to-parallel-loops  (current C++ pipeline — hypothesis: fails)" \
    "--convert-linalg-to-parallel-loops"

# ── Variant B: proposed fix ───────────────────────────────────────────────────
run_variant \
    "convert-linalg-to-loops  (sequential inner loops — hypothesis: emits kernel)" \
    "--convert-linalg-to-loops"

# ── Variant C: two-level transform (block + thread tiling) ───────────────────
# Uses tile_fuse_gpu_2level.mlir: level 1 tiles relu [32,32] for blocks and
# fuses producers; level 2 tiles the block-level relu [1,1] for threads and
# fuses producers again. Result: scf.forall[4,4] → scf.forall[32,32] nesting.
# After scf-forall-to-parallel on both, gpu-map-parallel-loops sees a clean
# blockIdx [4,4] → threadIdx [32,32] structure. Each thread does its own
# sequential K reduction for matmul — no sibling-parallel-loop ambiguity.
sep
echo "VARIANT C — two-level transform (block [32,32] + thread [1,1])"
echo "             goal: grid(4,4) block(32,32), 1024 threads per block"
sep

ir_c_stage1=""
if ir_c_stage1=$("$MLIR_OPT" \
        "${COMMON_PASSES[@]}" \
        "$INPUT_2LEVEL" 2>&1); then
    echo "--- scf.forall nesting after two-level transform + bufferize ---"
    echo "$ir_c_stage1" | grep -E "^\s*(func\.func|scf\.(parallel|for|forall))" || \
        echo "(grep matched nothing — showing raw)"
else
    red "[ERROR] two-level transform/bufferize stage failed"
    echo "$ir_c_stage1"
fi

sep
echo "VARIANT C — full GPU pipeline"
sep

ir_c=""
exit_c=0
ir_c=$("$MLIR_OPT" \
        "${COMMON_PASSES[@]}" \
        --convert-linalg-to-loops \
        "${GPU_PASSES[@]}" \
        "$INPUT_2LEVEL" 2>&1) || exit_c=$?

if [[ $exit_c -ne 0 ]]; then
    red "[ERROR] mlir-opt returned exit code $exit_c"
    echo "$ir_c"
else
    echo "$ir_c"
    sep
    if echo "$ir_c" | grep -qE "gpu\.(func|launch_func)\b"; then
        green "[OK]   gpu.func / gpu.launch_func found — kernel emitted"
        # Show the gpu.launch_func line so we can read grid/block dimensions
        echo ""
        echo "Launch details:"
        echo "$ir_c" | grep -E "gpu\.(launch_func|func)\b" | head -10
    else
        red "[FAIL] No gpu.func in output — no kernel emitted"
    fi
fi

# ── Variant D: two-level transform + parallel-loops ──────────────────────────
# Same two-level MLIR as Variant C but using convert-linalg-to-parallel-loops.
# The inner linalg ops are 1×1 tiles, so the resulting scf.parallel loops are
# [1,1] — they canonicalize away and should not block GPU mapping.
# This confirms whether the two-level approach is compatible with the parallel
# loops path (which avoids sequential K reduction ambiguity).
sep
echo "VARIANT D — two-level transform + convert-linalg-to-parallel-loops"
echo "             (1×1 inner parallels should canonicalize away)"
sep

ir_d=""
exit_d=0
ir_d=$("$MLIR_OPT" \
        "${COMMON_PASSES[@]}" \
        --convert-linalg-to-parallel-loops \
        --canonicalize \
        "${GPU_PASSES[@]}" \
        "$INPUT_2LEVEL" 2>&1) || exit_d=$?

if [[ $exit_d -ne 0 ]]; then
    red "[ERROR] mlir-opt returned exit code $exit_d"
    echo "$ir_d"
else
    echo "$ir_d"
    sep
    if echo "$ir_d" | grep -qE "gpu\.(func|launch_func)\b"; then
        green "[OK]   gpu.func / gpu.launch_func found — kernel emitted"
        echo ""
        echo "Launch details:"
        echo "$ir_d" | grep -E "gpu\.(launch_func|func)\b" | head -10
    else
        red "[FAIL] No gpu.func in output — no kernel emitted"
    fi
fi

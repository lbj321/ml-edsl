#!/usr/bin/env bash
# compile_gpu.sh — compile dense_gpu.mlir to PTX for RTX 2070 (sm_75)
#
# Usage:
#   chmod +x compile_gpu.sh
#   ./compile_gpu.sh                        # full pipeline to PTX
#   ./compile_gpu.sh --stop-after-tiling    # inspect tiled IR only
#   ./compile_gpu.sh --stop-after-bufferize # inspect bufferized IR
#   ./compile_gpu.sh --stop-after-gpu       # inspect gpu.launch IR

set -euo pipefail

INPUT="dense_gpu.mlir"
OUT_TILED="out_01_tiled.mlir"
OUT_BUFFERIZED="out_02_bufferized.mlir"
OUT_GPU="out_03_gpu_launch.mlir"
OUT_PTX="out_04_nvvm.mlir"

CHIP="sm_75"
PTX="ptx64"

# ── helpers ────────────────────────────────────────────────────────────────
die()  { echo "ERROR: $*" >&2; exit 1; }
info() { echo ">>> $*"; }

command -v mlir-opt &>/dev/null || die "mlir-opt not found in PATH"

STOP_AFTER=""
for arg in "$@"; do
  case "$arg" in
    --stop-after-tiling)    STOP_AFTER="tiling"    ;;
    --stop-after-bufferize) STOP_AFTER="bufferize" ;;
    --stop-after-gpu)       STOP_AFTER="gpu"       ;;
    *) die "unknown option: $arg" ;;
  esac
done

# ── Step 1: tiling + fusion (transform dialect) ────────────────────────────
info "Step 1: tiling + epilogue fusion"
mlir-opt "$INPUT" \
  --transform-interpreter \
  --canonicalize -cse \
  -o "$OUT_TILED"

if [[ "$STOP_AFTER" == "tiling" ]]; then
  info "Stopped after tiling. Output: $OUT_TILED"
  cat "$OUT_TILED"
  exit 0
fi

# ── Step 2: bufferize ──────────────────────────────────────────────────────
info "Step 2: bufferization"
mlir-opt "$OUT_TILED" \
  --one-shot-bufferize="bufferize-function-boundaries" \
  --canonicalize -cse \
  -o "$OUT_BUFFERIZED"

if [[ "$STOP_AFTER" == "bufferize" ]]; then
  info "Stopped after bufferization. Output: $OUT_BUFFERIZED"
  cat "$OUT_BUFFERIZED"
  exit 0
fi

# ── Step 3: GPU launch ────────────────────────────────────────────────────
info "Step 3: scf.forall → gpu.launch → kernel outline (sm_75)"
mlir-opt "$OUT_BUFFERIZED" \
  --pass-pipeline="builtin.module(
    func.func(
      linalg-generalize-named-ops,
      scf-forall-to-parallel,
      convert-linalg-to-loops,
      promote-buffers-to-stack{max-rank-of-allocated-memref=2},
      gpu-map-parallel-loops,
      convert-parallel-loops-to-gpu,
      lower-affine,
      convert-scf-to-cf,
      canonicalize,
      cse
    ),
    gpu-kernel-outlining,
    canonicalize
  )" \
  -o "$OUT_GPU"

if [[ "$STOP_AFTER" == "gpu" ]]; then
  info "Stopped after GPU lowering. Output: $OUT_GPU"
  cat "$OUT_GPU"
  exit 0
fi

# ── Step 4: NVVM lowering + host LLVM finalization ────────────────────────
info "Step 4: gpu → NVVM + host LLVM finalization"
mlir-opt "$OUT_GPU" \
  --pass-pipeline="builtin.module(
    gpu.module(
      convert-gpu-to-nvvm,
      reconcile-unrealized-casts
    ),
    expand-strided-metadata,
    convert-arith-to-llvm,
    convert-index-to-llvm,
    finalize-memref-to-llvm,
    convert-func-to-llvm,
    convert-cf-to-llvm,
    reconcile-unrealized-casts
  )" \
  -o "$OUT_PTX"

info "Done. Outputs:"
info "  $OUT_TILED       — after tiling/fusion"
info "  $OUT_BUFFERIZED  — after bufferization"
info "  $OUT_GPU         — after gpu.launch lowering"
info "  $OUT_PTX         — final LLVM IR with embedded PTX"

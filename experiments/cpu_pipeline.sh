#!/usr/bin/env bash
# Experiment: tensor-land → affine-based CPU pipeline
#
# Usage:
#   STEP=fused    ./cpu_pipeline.sh   # linalg fusion before bufferization
#   STEP=buffered ./cpu_pipeline.sh   # after one-shot bufferize
#   STEP=affine   ./cpu_pipeline.sh   # after linalg → affine loops
#   STEP=tiled    ./cpu_pipeline.sh   # after affine tiling
#   STEP=lowered  ./cpu_pipeline.sh   # after lower-affine → scf/cf
#   ./cpu_pipeline.sh                 # full → LLVM dialect

MLIR_OPT=/home/larsan/dev/llvm-project/build/bin/mlir-opt
INPUT=matmul_bias_relu.mlir
STEP=${STEP:-full}

run() {
  echo "=== $1 ===" >&2
  $MLIR_OPT "${@:2}"
}

# Step 1: fuse elementwise linalg ops at tensor level (bias+relu into matmul)
# This is the key step — fuses BEFORE K reduction is exposed
FUSED=$(run "linalg elementwise fusion (tensor level)" $INPUT \
  --linalg-fuse-elementwise-ops \
  --canonicalize)

[ "$STEP" = "fused" ] && { echo "$FUSED"; exit 0; }

# Step 2: one-shot bufferize (tensor → memref)
BUFFERED=$(echo "$FUSED" | run "one-shot bufferize" - \
  "--one-shot-bufferize=bufferize-function-boundaries=true function-boundary-type-conversion=identity-layout-map" \
  --canonicalize)

[ "$STEP" = "buffered" ] && { echo "$BUFFERED"; exit 0; }

# Step 3: linalg → affine loops (now in memref land)
AFFINE=$(echo "$BUFFERED" | run "linalg → affine" - \
  --convert-linalg-to-affine-loops \
  --canonicalize)

[ "$STEP" = "affine" ] && { echo "$AFFINE"; exit 0; }

# Step 4: tile affine loops for cache blocking
TILED=$(echo "$AFFINE" | run "affine tiling (8x8)" - \
  "--affine-loop-tile=tile-size=8" \
  --canonicalize)

[ "$STEP" = "tiled" ] && { echo "$TILED"; exit 0; }

# Step 5: lower affine → scf/cf
LOWERED=$(echo "$TILED" | run "lower-affine → scf/cf" - \
  --lower-affine \
  --canonicalize)

[ "$STEP" = "lowered" ] && { echo "$LOWERED"; exit 0; }

# Step 6: lower to LLVM dialect
echo "$LOWERED" | run "lower to LLVM" - \
  --convert-vector-to-llvm \
  --convert-arith-to-llvm \
  --finalize-memref-to-llvm \
  --convert-cf-to-llvm \
  --convert-func-to-llvm \
  --reconcile-unrealized-casts

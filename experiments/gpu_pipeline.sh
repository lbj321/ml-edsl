#!/usr/bin/env bash
# Experiment: GPU pipeline for matmul (linalg → scf.forall → gpu.launch)
# Stops at each stage to inspect IR.
#
# Usage:
#   ./gpu_pipeline.sh               # full pipeline → NVVM IR
#   STEP=parallel ./gpu_pipeline.sh # stop after linalg → scf.parallel
#   STEP=gpu      ./gpu_pipeline.sh # stop after scf.parallel → gpu.launch
#   STEP=outlined ./gpu_pipeline.sh # stop after kernel outlining

MLIR_OPT=/home/larsan/dev/llvm-project/build/bin/mlir-opt
INPUT=matmul_bias_relu.mlir
STEP=${STEP:-full}

run() {
  echo "=== $1 ===" >&2
  $MLIR_OPT "${@:2}"
}

# Step 1: linalg → scf.parallel (tile + map)
PARALLEL=$(run "linalg → scf.parallel" $INPUT \
  --convert-linalg-to-parallel-loops \
  --gpu-map-parallel-loops \
  --canonicalize)

[ "$STEP" = "parallel" ] && { echo "$PARALLEL"; exit 0; }

# Step 2: scf.parallel → gpu.launch
GPU=$(echo "$PARALLEL" | run "scf.parallel → gpu.launch" - \
  --convert-parallel-loops-to-gpu \
  --canonicalize)

[ "$STEP" = "gpu" ] && { echo "$GPU"; exit 0; }

# Step 3: outline gpu.launch bodies to gpu.func / gpu.module
OUTLINED=$(echo "$GPU" | run "kernel outlining" - \
  --gpu-kernel-outlining \
  --canonicalize)

[ "$STEP" = "outlined" ] && { echo "$OUTLINED"; exit 0; }

# Step 4: lower gpu.module contents to NVVM dialect
echo "$OUTLINED" | run "lower to NVVM" - \
  --convert-gpu-to-nvvm \
  --lower-affine \
  --convert-arith-to-llvm \
  --finalize-memref-to-llvm \
  --convert-cf-to-llvm \
  --convert-func-to-llvm \
  --reconcile-unrealized-casts

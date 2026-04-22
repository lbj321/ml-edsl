#!/usr/bin/env bash
# Experiment: linalg → scf.parallel → async CPU parallelism pipeline
#
# Usage:
#   ./async_pipeline.sh                   # full → LLVM dialect
#   STEP=buffered ./async_pipeline.sh     # after one-shot bufferize
#   STEP=parallel ./async_pipeline.sh     # after linalg → scf.parallel
#   STEP=async    ./async_pipeline.sh     # after scf.parallel → async.execute
#   STEP=runtime  ./async_pipeline.sh     # after async → async.runtime ops

MLIR_OPT=/home/larsan/dev/llvm-project/build/bin/mlir-opt
INPUT=matmul_bias_relu.mlir
STEP=${STEP:-full}

run() {
  echo "=== $1 ===" >&2
  $MLIR_OPT "${@:2}"
}

# Step 1: one-shot bufferize (tensor → memref)
BUFFERED=$(run "one-shot bufferize" $INPUT \
  "--one-shot-bufferize=bufferize-function-boundaries=true function-boundary-type-conversion=identity-layout-map" \
  --canonicalize)

[ "$STEP" = "buffered" ] && { echo "$BUFFERED"; exit 0; }

# Step 2: linalg → scf.parallel (all loops: i,j parallel; k serial inner loop for matmul)
PARALLEL=$(echo "$BUFFERED" | run "linalg → scf.parallel" - \
  --convert-linalg-to-parallel-loops \
  --canonicalize)

[ "$STEP" = "parallel" ] && { echo "$PARALLEL"; exit 0; }

# Step 3: scf.parallel → async.execute tasks
ASYNC=$(echo "$PARALLEL" | run "scf.parallel → async.execute" - \
  --async-parallel-for \
  --canonicalize)

[ "$STEP" = "async" ] && { echo "$ASYNC"; exit 0; }

# Step 4: async.execute → explicit async.runtime ops + coroutines
RUNTIME=$(echo "$ASYNC" | run "async → async.runtime" - \
  --async-to-async-runtime \
  --async-func-to-async-runtime \
  --async-runtime-ref-counting \
  --async-runtime-ref-counting-opt \
  --canonicalize)

[ "$STEP" = "runtime" ] && { echo "$RUNTIME"; exit 0; }

# Step 5: lower to LLVM dialect
echo "$RUNTIME" | run "lower to LLVM" - \
  --convert-async-to-llvm \
  --convert-vector-to-llvm \
  --convert-arith-to-llvm \
  --finalize-memref-to-llvm \
  --convert-cf-to-llvm \
  --convert-func-to-llvm \
  --reconcile-unrealized-casts

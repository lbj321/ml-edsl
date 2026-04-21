#!/bin/bash
set -e
MLIR_OPT=${MLIR_OPT:-mlir-opt}

echo "=== Step 1: bufferize + linalg-to-parallel-loops ==="
$MLIR_OPT ir_output/matmul_gpu.mlir \
  --linalg-fuse-elementwise-ops \
  --one-shot-bufferize="bufferize-function-boundaries=true function-boundary-type-conversion=identity-layout-map" \
  --ownership-based-buffer-deallocation \
  --canonicalize \
  --convert-linalg-to-parallel-loops \
  > /tmp/gpu_step1.mlir
echo "OK"

echo "=== Step 2: gpu-map-parallel-loops ==="
$MLIR_OPT /tmp/gpu_step1.mlir --gpu-map-parallel-loops > /tmp/gpu_step2.mlir
echo "OK"
cat /tmp/gpu_step2.mlir

echo "=== Step 3: convert-parallel-loops-to-gpu + gpu-kernel-outlining ==="
$MLIR_OPT /tmp/gpu_step2.mlir --convert-parallel-loops-to-gpu --gpu-kernel-outlining --canonicalize > /tmp/gpu_step3.mlir
echo "OK"
cat /tmp/gpu_step3.mlir

echo "=== Step 4: lower gpu.module to NVVM ==="
$MLIR_OPT /tmp/gpu_step3.mlir \
  --pass-pipeline="builtin.module(gpu.module(convert-gpu-to-nvvm),convert-scf-to-cf,expand-strided-metadata,lower-affine,convert-arith-to-llvm,finalize-memref-to-llvm,convert-cf-to-llvm,reconcile-unrealized-casts)" \
  > /tmp/gpu_step4.mlir
echo "OK"
cat /tmp/gpu_step4.mlir

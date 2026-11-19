#!/usr/bin/env bash
# Stage 1 end-to-end: transform script -> bufferize -> scalarize A's packed
# read -> lower to LLVM -> opt -O3 -> llc -O3. Every step here was load-
# bearing (see PLAN.md's Stage 1 section for what happens if you skip one):
#
#   - `-test-scalar-vector-transfer-lowering=allow-multiple-uses` turns the
#     one <6 x float> tile-read of A into 6 independent scalar loads. Without
#     it, A's illegal 24-byte vector type gets split by the legalizer into a
#     16-byte + 8-byte load, and the 8-byte piece needs an extra vpermps
#     (with a spilled shuffle-control constant) to pick out element 5.
#   - `-buffer-hoisting -buffer-loop-hoisting` right after
#     `-convert-vector-to-scf` hoists the temporary allocas that pass
#     introduces for multi-dim vector.transfer ops OUT of the k-loop. Skip
#     it and every iteration does a dynamic stack realignment.
#   - `opt -passes='default<O3>' -mcpu=skylake` between mlir-translate and
#     llc is the big one: llc alone only does codegen (no SROA/InstCombine/
#     LICM/LoopRotate), and handing it an unrotated loop with an aggregate
#     [6 x <16 x float>] loop-carried phi produces a register-rotation
#     shuffle chain plus real accumulator spills. `opt -O3` first fixes
#     this completely -- SelectionDAG splits the (still-aggregate-typed)
#     phi into 12 independent per-element virtual registers once the loop
#     is in good shape.
#
# mlir-opt/mlir-translate/opt/llc/llvm-mca must be on PATH or pointed at
# via MLIR_OPT/MLIR_TRANSLATE/LLVM_OPT/LLVM_LLC/LLVM_MCA.
set -euo pipefail

MLIR_OPT="${MLIR_OPT:-mlir-opt}"
MLIR_TRANSLATE="${MLIR_TRANSLATE:-mlir-translate}"
LLVM_OPT="${LLVM_OPT:-opt}"
LLVM_LLC="${LLVM_LLC:-llc}"
LLVM_MCA="${LLVM_MCA:-llvm-mca}"

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="$DIR/../out"
mkdir -p "$OUT"

echo "== transform: tile K by 1, vectorize, lower contraction/outerproduct, hoist =="
"$MLIR_OPT" "$DIR/stage1_microkernel.mlir" \
  -transform-preload-library="transform-library-paths=$DIR/stage1_transform.mlir" \
  -transform-interpreter \
  -canonicalize -cse \
  -o "$OUT/stage1_vectorized.mlir"

echo "== bufferize =="
"$MLIR_OPT" "$OUT/stage1_vectorized.mlir" \
  -eliminate-empty-tensors \
  -one-shot-bufferize="bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map" \
  -canonicalize -cse \
  -o "$OUT/stage1_bufferized.mlir"

echo "== scalarize A's packed tile-read into 6 independent scalar loads =="
"$MLIR_OPT" "$OUT/stage1_bufferized.mlir" \
  -test-scalar-vector-transfer-lowering=allow-multiple-uses \
  -canonicalize -cse \
  -o "$OUT/stage1_scalarized.mlir"

echo "== lower to LLVM dialect (with alloca hoisting after convert-vector-to-scf) =="
"$MLIR_OPT" "$OUT/stage1_scalarized.mlir" \
  -convert-vector-to-scf \
  -buffer-hoisting -buffer-loop-hoisting \
  -canonicalize \
  -convert-vector-to-llvm="enable-x86vector" \
  -convert-ub-to-llvm \
  -convert-scf-to-cf \
  -expand-strided-metadata \
  -lower-affine \
  -convert-arith-to-llvm \
  -finalize-memref-to-llvm \
  -convert-cf-to-llvm \
  -convert-func-to-llvm \
  -reconcile-unrealized-casts \
  -o "$OUT/stage1_llvm.mlir"

echo "== translate + optimize + codegen =="
"$MLIR_TRANSLATE" --mlir-to-llvmir "$OUT/stage1_llvm.mlir" -o "$OUT/stage1.ll"
"$LLVM_OPT" -passes='default<O3>' -mcpu=skylake "$OUT/stage1.ll" -S -o "$OUT/stage1.opt.ll"
"$LLVM_LLC" -O3 -mcpu=skylake -relocation-model=pic "$OUT/stage1.opt.ll" -o "$OUT/stage1.opt.s"
"$LLVM_LLC" -mcpu=skylake -O3 -relocation-model=pic -filetype=obj "$OUT/stage1.opt.ll" -o "$OUT/stage1_kernel.o"
clang -shared -fPIC "$OUT/stage1_kernel.o" -o "$OUT/libstage1_kernel.so"

echo "== assembly checks (loop body) =="
awk '/^\.LBB0_1:/,/^# %bb\.2:/' "$OUT/stage1.opt.s" | grep -v "^# %bb\.2:" > "$OUT/stage1_loop_body.s"
echo "vfmadd231ps: $(grep -c vfmadd231ps "$OUT/stage1_loop_body.s") (want 12)"
echo "vbroadcastss: $(grep -c vbroadcastss "$OUT/stage1_loop_body.s") (want 6)"
echo "vmovups: $(grep -c vmovups "$OUT/stage1_loop_body.s") (want 2)"
echo "spill/reload: $(grep -c 'Spill\|Reload' "$OUT/stage1_loop_body.s") (want 0)"

echo "== llvm-mca (FMA-port-bound check) =="
"$LLVM_MCA" -mcpu=skylake -iterations=100 -resource-pressure "$OUT/stage1_loop_body.s" \
  | sed -n '1,10p;/^Resource pressure per iteration/,/^$/p'

echo "wrote $OUT/libstage1_kernel.so (symbol: microkernel_6x16)"

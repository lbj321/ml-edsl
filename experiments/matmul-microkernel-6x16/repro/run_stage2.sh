#!/usr/bin/env bash
# Stage 2 end-to-end: same recipe as run_stage1.sh (bufferize -> scalarize
# A's packed read -> lower to LLVM -> opt -O3 -> llc -O3), applied to the
# macro-kernel (jr/ir loops wrapped around the Stage 1 microkernel) instead
# of the bare kernel. The point of Stage 2 is to check that wrapping the
# kernel in loops didn't change its assembly -- so this script diffs the
# innermost loop body's instruction counts against Stage 1's.
set -euo pipefail

MLIR_OPT="${MLIR_OPT:-mlir-opt}"
MLIR_TRANSLATE="${MLIR_TRANSLATE:-mlir-translate}"
LLVM_OPT="${LLVM_OPT:-opt}"
LLVM_LLC="${LLVM_LLC:-llc}"
LLVM_MCA="${LLVM_MCA:-llvm-mca}"

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="$DIR/../out"
mkdir -p "$OUT"

echo "== transform: jr (N, NR=16) -> ir (M, MR=6) -> Stage 1's K-tile/vectorize/hoist recipe =="
"$MLIR_OPT" "$DIR/stage2_macrokernel.mlir" \
  -transform-preload-library="transform-library-paths=$DIR/stage2_transform.mlir" \
  -transform-interpreter \
  -canonicalize -cse \
  -o "$OUT/stage2_vectorized.mlir"

echo "== bufferize =="
"$MLIR_OPT" "$OUT/stage2_vectorized.mlir" \
  -eliminate-empty-tensors \
  -one-shot-bufferize="bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map" \
  -canonicalize -cse \
  -o "$OUT/stage2_bufferized.mlir"

echo "== scalarize A's packed tile-read into 6 independent scalar loads =="
"$MLIR_OPT" "$OUT/stage2_bufferized.mlir" \
  -test-scalar-vector-transfer-lowering=allow-multiple-uses \
  -canonicalize -cse \
  -o "$OUT/stage2_scalarized.mlir"

echo "== lower to LLVM dialect (with alloca hoisting after convert-vector-to-scf) =="
"$MLIR_OPT" "$OUT/stage2_scalarized.mlir" \
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
  -o "$OUT/stage2_llvm.mlir"

echo "== translate + optimize + codegen =="
"$MLIR_TRANSLATE" --mlir-to-llvmir "$OUT/stage2_llvm.mlir" -o "$OUT/stage2.ll"
"$LLVM_OPT" -passes='default<O3>' -mcpu=skylake "$OUT/stage2.ll" -S -o "$OUT/stage2.opt.ll"
"$LLVM_LLC" -O3 -mcpu=skylake -relocation-model=pic "$OUT/stage2.opt.ll" -o "$OUT/stage2.opt.s"
"$LLVM_LLC" -mcpu=skylake -O3 -relocation-model=pic -filetype=obj "$OUT/stage2.opt.ll" -o "$OUT/stage2_kernel.o"
clang -shared -fPIC "$OUT/stage2_kernel.o" -o "$OUT/libstage2_kernel.so"

echo "== assembly checks (innermost k-loop body -- should match Stage 1 exactly) =="
# Grab the block containing vfmadd231ps, bounded to just its own loop (label
# to its own backedge jump), not the tile load/store code that follows it.
python3 - "$OUT/stage2.opt.s" "$OUT/stage2_loop_body.s" << 'PYEOF'
import re, sys
src, dst = sys.argv[1], sys.argv[2]
lines = open(src).read().splitlines()
label = None
for i, line in enumerate(lines):
    m = re.match(r'(\.LBB0_\d+):', line)
    if m:
        label = m.group(1)
    if "vfmadd231ps" in line and label:
        start = next(j for j in range(i, -1, -1) if lines[j].startswith(label + ":"))
        end = next(j for j in range(i, len(lines)) if re.search(r'\bj\w+\s+' + re.escape(label) + r'\b', lines[j]))
        open(dst, "w").write("\n".join(lines[start:end + 1]) + "\n")
        break
PYEOF
echo "vfmadd231ps: $(grep -c vfmadd231ps "$OUT/stage2_loop_body.s") (want 12, same as Stage 1)"
echo "vbroadcastss: $(grep -c vbroadcastss "$OUT/stage2_loop_body.s") (want 6)"
echo "vmovups: $(grep -c vmovups "$OUT/stage2_loop_body.s") (want 2)"
echo "spill/reload: $(grep -c 'Spill\|Reload' "$OUT/stage2_loop_body.s") (want 0)"

echo "== llvm-mca (should match Stage 1's ~6.14 cycles/iter) =="
"$LLVM_MCA" -mcpu=skylake -iterations=100 -resource-pressure "$OUT/stage2_loop_body.s" \
  | sed -n '1,10p;/^Resource pressure per iteration/,/^$/p'

echo "wrote $OUT/libstage2_kernel.so (symbol: macrokernel_168x64)"

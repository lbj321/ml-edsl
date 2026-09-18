#!/usr/bin/env bash
# Stage 3 end-to-end: pack A/B globally (approach 1), tile through all five
# loop levels (jc -> pc -> ic -> jr -> ir), apply Stage 1's exact
# K-tile/vectorize/lower/hoist recipe to the innermost 6x16xKC tile, then
# lower_pack/lower_unpack + tile + vectorize the packing transposes.
# Same bufferize -> scalarize -> lower -> opt -> llc recipe as run_stage1.sh/
# run_stage2.sh, applied to the full outer-loop nest.
set -euo pipefail

MLIR_OPT="${MLIR_OPT:-mlir-opt}"
MLIR_TRANSLATE="${MLIR_TRANSLATE:-mlir-translate}"
LLVM_OPT="${LLVM_OPT:-opt}"
LLVM_LLC="${LLVM_LLC:-llc}"
LLVM_MCA="${LLVM_MCA:-llvm-mca}"

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="$DIR/../out"
mkdir -p "$OUT"

echo "== transform: pack A/B -> interchange -> jc/pc/ic/jr/ir/k tiling -> vectorize -> contract -> outerproduct -> hoist -> lower_pack/unpack -> tile+vectorize transposes =="
"$MLIR_OPT" "$DIR/stage3_outer.mlir" \
  -transform-preload-library="transform-library-paths=$DIR/stage3_transform.mlir" \
  -transform-interpreter \
  -canonicalize -cse \
  -o "$OUT/stage3_vectorized.mlir" 2> "$OUT/stage3_transform_log.txt"

echo "== bufferize =="
"$MLIR_OPT" "$OUT/stage3_vectorized.mlir" \
  -eliminate-empty-tensors \
  -one-shot-bufferize="bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map" \
  -canonicalize -cse \
  -o "$OUT/stage3_bufferized.mlir"

echo "== scalarize A's packed tile-read into 6 independent scalar loads =="
"$MLIR_OPT" "$OUT/stage3_bufferized.mlir" \
  -test-scalar-vector-transfer-lowering=allow-multiple-uses \
  -canonicalize -cse \
  -o "$OUT/stage3_scalarized.mlir"

echo "== lower to LLVM dialect (with alloca hoisting after convert-vector-to-scf) =="
# Stage 3, unlike Stage 1/2, still has two linalg ops after scalarization
# (linalg.fill zero-initing the packed-C buffer, linalg.copy for the final
# C unpack) -- neither is tiled/vectorized (the fill because it inits the
# whole packed buffer before the compute loop nest, the copy because C's
# unpack transpose degenerated to a pure identity permutation that
# canonicalized straight to extract_slice/insert_slice, bypassing
# vectorize entirely). Both must go through convert-linalg-to-loops before
# the LLVM dialect conversions, or mlir-translate fails with "Dialect
# 'linalg' not found".
"$MLIR_OPT" "$OUT/stage3_scalarized.mlir" \
  -convert-vector-to-scf \
  -convert-linalg-to-loops \
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
  -o "$OUT/stage3_llvm.mlir"

echo "== translate + optimize + codegen =="
"$MLIR_TRANSLATE" --mlir-to-llvmir "$OUT/stage3_llvm.mlir" -o "$OUT/stage3.ll"
"$LLVM_OPT" -passes='default<O3>' -mcpu=skylake "$OUT/stage3.ll" -S -o "$OUT/stage3.opt.ll"
"$LLVM_LLC" -O3 -mcpu=skylake -relocation-model=pic "$OUT/stage3.opt.ll" -o "$OUT/stage3.opt.s"
"$LLVM_LLC" -mcpu=skylake -O3 -relocation-model=pic -filetype=obj "$OUT/stage3.opt.ll" -o "$OUT/stage3_kernel.o"
clang -shared -fPIC "$OUT/stage3_kernel.o" -o "$OUT/libstage3_kernel.so"

echo "== assembly checks (innermost k-loop body -- should match Stage 1/2 exactly) =="
python3 - "$OUT/stage3.opt.s" "$OUT/stage3_loop_body.s" << 'PYEOF'
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
echo "vfmadd231ps: $(grep -c vfmadd231ps "$OUT/stage3_loop_body.s") (want 12, same as Stage 1/2)"
echo "vbroadcastss: $(grep -c vbroadcastss "$OUT/stage3_loop_body.s") (want 6)"
# B's packed panel loads as vmovaps here, not vmovups like Stage 1/2 --
# LLVM proved 32-byte alignment on the packed buffer, so it picked the
# aligned mnemonic. Same instruction count and semantics either way.
echo "vmovups+vmovaps (B loads): $(grep -cE 'vmovups|vmovaps' "$OUT/stage3_loop_body.s") (want 2)"
echo "spill/reload: $(grep -c 'Spill\|Reload' "$OUT/stage3_loop_body.s") (want 0)"

echo "== llvm-mca (should match Stage 1/2's ~6.14 cycles/iter) =="
"$LLVM_MCA" -mcpu=skylake -iterations=100 -resource-pressure "$OUT/stage3_loop_body.s" \
  | sed -n '1,10p;/^Resource pressure per iteration/,/^$/p'

echo "wrote $OUT/libstage3_kernel.so (symbol: outer_336x64x512)"

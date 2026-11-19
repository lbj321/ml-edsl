#!/usr/bin/env bash
# Stage 3, approach 2 (BLIS per-block packing): tile the named matmul through
# jc -> pc -> ic -> jr -> ir, pad the 6x16xKC microtile's A/B operands and
# hoist_pad them out to the pc (B~) / ic (A~) loops, then Stage 2's
# K-tile/vectorize/lower/hoist recipe on the microtile. See
# stage3b_transform.mlir. Same bufferize -> scalarize -> lower -> opt -> llc
# recipe as run_stage3.sh (including Fix 7, full-unroll vector-to-scf).
#
# Env overrides: INPUT (default stage3_outer.mlir), TRANSFORM (default
# stage3b_transform.mlir), TAG (output-name prefix, default stage3b).
set -euo pipefail

MLIR_OPT="${MLIR_OPT:-mlir-opt}"
MLIR_TRANSLATE="${MLIR_TRANSLATE:-mlir-translate}"
LLVM_OPT="${LLVM_OPT:-opt}"
LLVM_LLC="${LLVM_LLC:-llc}"
LLVM_MCA="${LLVM_MCA:-llvm-mca}"

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="$DIR/../out"
mkdir -p "$OUT"
INPUT="${INPUT:-$DIR/stage3_outer.mlir}"
TRANSFORM="${TRANSFORM:-$DIR/stage3b_transform.mlir}"
TAG="${TAG:-stage3b}"

echo "== transform: jc/pc/ic/jr/ir tiling -> pad -> hoist_pad B~ (pc) / A~ (ic) -> k-tile -> vectorize -> contract -> outerproduct -> hoist =="
"$MLIR_OPT" "$INPUT" \
  -transform-preload-library="transform-library-paths=$TRANSFORM" \
  -transform-interpreter \
  -canonicalize -cse \
  -o "$OUT/${TAG}_vectorized.mlir" > "$OUT/${TAG}_transform_log.txt" 2>&1

echo "== bufferize =="
"$MLIR_OPT" "$OUT/${TAG}_vectorized.mlir" \
  -eliminate-empty-tensors \
  -one-shot-bufferize="bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map" \
  -canonicalize -cse \
  -o "$OUT/${TAG}_bufferized.mlir"

echo "== scalarize A's packed tile-read into 6 independent scalar loads =="
"$MLIR_OPT" "$OUT/${TAG}_bufferized.mlir" \
  -test-scalar-vector-transfer-lowering=allow-multiple-uses \
  -canonicalize -cse \
  -o "$OUT/${TAG}_scalarized.mlir"

echo "== lower to LLVM dialect (with alloca hoisting after convert-vector-to-scf) =="
# linalg.fill (zero-init of C) is the only linalg op left -- lower it with
# convert-linalg-to-loops.
#
# Fix 7: convert-vector-to-scf needs full-unroll=true. Its default
# (progressive) lowering of the n-D transfers left here -- the 6x16 C
# microtile read/write around each k-loop, and the 8x6/8x16 packing-tile
# transfers -- goes through a stack temp buffer (memref<vector<...>>); on
# these strided subviews opt -O3 couldn't SROA it away, leaving 12
# `callq memcpy` (384 B in + out per microtile, 512 B per B-pack
# iteration). full-unroll emits one 1-D transfer per row instead: memcpy
# count 0, kernel loop unchanged. (Dropping the unit dims from the rank-4
# C-tile / 8x1x16 B-pack transfers was tried first and made no difference
# on its own -- the temp buffer is used for every n-D transfer.)
"$MLIR_OPT" "$OUT/${TAG}_scalarized.mlir" \
  -convert-vector-to-scf="full-unroll=true" \
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
  -o "$OUT/${TAG}_llvm.mlir"

echo "== translate + optimize + codegen =="
"$MLIR_TRANSLATE" --mlir-to-llvmir "$OUT/${TAG}_llvm.mlir" -o "$OUT/${TAG}.ll"
"$LLVM_OPT" -passes='default<O3>' -mcpu=skylake "$OUT/${TAG}.ll" -S -o "$OUT/${TAG}.opt.ll"
"$LLVM_LLC" -O3 -mcpu=skylake -relocation-model=pic "$OUT/${TAG}.opt.ll" -o "$OUT/${TAG}.opt.s"
"$LLVM_LLC" -mcpu=skylake -O3 -relocation-model=pic -filetype=obj "$OUT/${TAG}.opt.ll" -o "$OUT/${TAG}_kernel.o"
clang -shared -fPIC "$OUT/${TAG}_kernel.o" -o "$OUT/lib${TAG}_kernel.so"

echo "== assembly checks (innermost k-loop body -- should match Stage 1/2 exactly) =="
python3 - "$OUT/${TAG}.opt.s" "$OUT/${TAG}_loop_body.s" << 'PYEOF'
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
echo "vfmadd231ps: $(grep -c vfmadd231ps "$OUT/${TAG}_loop_body.s") (want 12, same as Stage 1/2)"
echo "vbroadcastss: $(grep -c vbroadcastss "$OUT/${TAG}_loop_body.s") (want 6)"
# B~ loads may show as vmovaps, not vmovups like Stage 1/2 --
# LLVM proved 32-byte alignment on the packed buffer, so it picked the
# aligned mnemonic. Same instruction count and semantics either way.
echo "vmovups+vmovaps (B loads): $(grep -cE 'vmovups|vmovaps' "$OUT/${TAG}_loop_body.s") (want 2)"
echo "spill/reload: $(grep -c 'Spill\|Reload' "$OUT/${TAG}_loop_body.s") (want 0)"
echo "memcpy calls (whole function): $(grep -c 'memcpy' "$OUT/${TAG}.opt.s" || true) (want 0)"

echo "== llvm-mca (should match Stage 1/2's ~6.14 cycles/iter) =="
"$LLVM_MCA" -mcpu=skylake -iterations=100 -resource-pressure "$OUT/${TAG}_loop_body.s" \
  | sed -n '1,10p;/^Resource pressure per iteration/,/^$/p'

echo "wrote $OUT/lib${TAG}_kernel.so"

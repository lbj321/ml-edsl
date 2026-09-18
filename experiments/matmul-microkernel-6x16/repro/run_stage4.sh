#!/usr/bin/env bash
# Stage 4: bufferization cleanup on top of Stage 3 approach 2. Reuses
# stage3b_transform.mlir unchanged; only the bufferize step differs from
# run_stage3b.sh:
#
#   - The packed-buffer allocs (A~ inside ic, B~ inside pc) are hoisted to
#     function entry right after bufferization (buffer-hoisting +
#     buffer-loop-hoisting), instead of as a side effect of flags placed late
#     in the LLVM-lowering step.
#   - Only then is ownership-based deallocation inserted, so it sees
#     function-level allocs and emits one dealloc per buffer before return.
#     Stage 3 never freed A~/B~ (one-shot-bufferize without deallocation):
#     every call leaked NC/16*KC*16 + MC/6*KC*6 floats.
#
# Structural checks at the end FAIL the script (exit 1), not just echo.
#
# Env overrides: INPUT (default stage3_outer.mlir), TRANSFORM (default
# stage3b_transform.mlir), TAG (output-name prefix, default stage4).
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
TAG="${TAG:-stage4}"

echo "== transform (stage3b_transform.mlir, unchanged) =="
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

echo "== hoist packed-buffer allocs, then insert deallocation =="
# Order matters: hoist first, so ownership-based deallocation sees
# function-level allocs and places a single dealloc before return, rather
# than an alloc/free pair inside the ic/pc loops. The leading -canonicalize
# also removes tiling's `memref.copy %x, %x` self-copies (FoldSelfCopy), which
# the bufferize step's trailing -cse exposes but doesn't clean up.
"$MLIR_OPT" "$OUT/${TAG}_bufferized.mlir" \
  -canonicalize \
  -buffer-hoisting -buffer-loop-hoisting \
  -ownership-based-buffer-deallocation \
  -canonicalize \
  -buffer-deallocation-simplification \
  -bufferization-lower-deallocations \
  -canonicalize -cse \
  -o "$OUT/${TAG}_dealloc.mlir"

echo "== scalarize A~'s packed tile-read into 6 independent scalar loads =="
"$MLIR_OPT" "$OUT/${TAG}_dealloc.mlir" \
  -test-scalar-vector-transfer-lowering=allow-multiple-uses \
  -canonicalize -cse \
  -o "$OUT/${TAG}_scalarized.mlir"

echo "== lower to LLVM dialect =="
# Same as run_stage3b.sh minus -buffer-hoisting/-buffer-loop-hoisting (now
# done right after bufferization). full-unroll=true is Stage 3's Fix 7.
"$MLIR_OPT" "$OUT/${TAG}_scalarized.mlir" \
  -convert-vector-to-scf="full-unroll=true" \
  -convert-linalg-to-loops \
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

echo "== checks =="
python3 - "$OUT/${TAG}_dealloc.mlir" "$OUT/${TAG}.opt.s" "$OUT/${TAG}_loop_body.s" << 'PYEOF'
import re, sys
mlir_path, asm_path, body_path = sys.argv[1:4]
failures = []

def check(name, got, want):
    ok = got == want
    print(f"{'ok  ' if ok else 'FAIL'} {name}: {got} (want {want})")
    if not ok:
        failures.append(name)

# --- MLIR, after hoisting + deallocation ---
mlir = open(mlir_path).read().splitlines()
copies = [l for l in mlir if "memref.copy" in l]
real_copies = [l for l in copies
               if (m := re.search(r"memref\.copy (%\w+), (%\w+)", l)) and m.group(1) != m.group(2)]
check("memref.copy (non-self) after bufferization", len(real_copies), 0)
check("memref.copy (self, should be canonicalized away)", len(copies) - len(real_copies), 0)
check("bufferization.dealloc left unlowered", sum("bufferization.dealloc" in l for l in mlir), 0)

# Function-body ops are indented 4 spaces (module > func); anything deeper is
# inside a loop/region.
def depth(l): return len(l) - len(l.lstrip(" "))
allocs = [l for l in mlir if "memref.alloc(" in l or "memref.alloc()" in l]
deallocs = [l for l in mlir if "memref.dealloc" in l]
check("memref.alloc count", len(allocs), 2)
check("memref.dealloc count", len(deallocs), 2)
check("allocs/deallocs not at function top level", sum(depth(l) != 4 for l in allocs + deallocs), 0)

# --- asm: whole function ---
asm = open(asm_path).read().splitlines()
# Count tail calls too: the last free before return is emitted as
# `jmp free@PLT  # TAILCALL`, not callq.
calls = [m.group(1) for l in asm if (m := re.match(r"\s+(?:callq|jmp)\s+(\S+@PLT)", l))]
check("malloc calls", sum(c.startswith("malloc") for c in calls), 2)
check("free calls", sum(c.startswith("free") for c in calls), 2)
check("memcpy calls", sum(c.startswith("memcpy") for c in calls), 0)
check("memrefCopy calls", sum("memrefCopy" in c for c in calls), 0)

# --- asm: innermost k-loop (the block containing the first FMA, from its
# label to its own backedge) -- must be identical to Stages 1/2/3 ---
label = None
body = None
for i, line in enumerate(asm):
    m = re.match(r'(\.LBB0_\d+):', line)
    if m:
        label = m.group(1)
    if "vfmadd231ps" in line and label:
        start = next(j for j in range(i, -1, -1) if asm[j].startswith(label + ":"))
        end = next(j for j in range(i, len(asm)) if re.search(r'\bj\w+\s+' + re.escape(label) + r'\b', asm[j]))
        body = asm[start:end + 1]
        break
if body is None:
    failures.append("kernel loop not found")
    body = []
open(body_path, "w").write("\n".join(body) + "\n")
check("kernel vfmadd231ps", sum("vfmadd231ps" in l for l in body), 12)
check("kernel vbroadcastss", sum("vbroadcastss" in l for l in body), 6)
check("kernel B loads (vmovups/vmovaps)", sum(bool(re.search(r"vmov[au]ps", l)) for l in body), 2)
check("kernel spills/reloads", sum(("Spill" in l or "Reload" in l) for l in body), 0)

if failures:
    print(f"{len(failures)} check(s) FAILED: {', '.join(failures)}")
    sys.exit(1)
print("all checks passed")
PYEOF

echo "== llvm-mca (kernel loop; expect ~6.14 cycles/iter) =="
"$LLVM_MCA" -mcpu=skylake -iterations=100 "$OUT/${TAG}_loop_body.s" | sed -n '1,4p'

echo "wrote $OUT/lib${TAG}_kernel.so"

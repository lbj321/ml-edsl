#!/usr/bin/env bash
# INTEGRATION.md Step 1 end-to-end: the blocked matmul as it exists in the
# compiler, driven through mlir-edsl-opt rather than a transform script.
#
# Unlike run_stage1..6.sh, which drive mlir-opt with a transform-dialect
# script, everything here comes from compiled passes:
#
#   -linalg-matmul-blocked   LinalgMatmulBlockedPass  (tiling: jc/pc/ic/jr/ir/k)
#   -cpu-pipeline    buildCPUPipeline(hoistLoopInvariantSubsets=true)
#
# The microkernel itself is built by passes the pipeline already had:
# linalg-matmul-to-contract -> linalg-vectorize ->
# loop-invariant-subset-hoisting -> vector-contract-to-outerproduct.
#
# No packing yet (Step 2), so A is read as MR scalars from MR rows and B is
# read with stride N between k steps. Expect well under the ~100 GFLOPS the
# fully packed Stage 4 script reached.
#
# Usage: ./run_step1.sh [M N K] [-- extra -linalg-matmul-blocked options]
#   e.g. ./run_step1.sh 1024 1024 1024
#        ./run_step1.sh 768 768 768 -- mr=6
set -euo pipefail

MLIR_TRANSLATE="${MLIR_TRANSLATE:-mlir-translate}"
LLVM_OPT="${LLVM_OPT:-opt}"
LLVM_LLC="${LLVM_LLC:-llc}"
MCPU="${MCPU:-skylake}"

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$DIR/../../.." && pwd)"
OUT="$DIR/../out"
EDSL_OPT="${EDSL_OPT:-$REPO/build/cpp/tools/mlir-edsl-opt/mlir-edsl-opt}"
mkdir -p "$OUT"

M="${1:-1024}"; N="${2:-1024}"; K="${3:-1024}"
shift 3 2>/dev/null || true
[ "${1:-}" = "--" ] && shift
BLOCK_OPTS="${*:-}"
PASS="-linalg-matmul-blocked"
[ -n "$BLOCK_OPTS" ] && PASS="-linalg-matmul-blocked=${BLOCK_OPTS// /,}"

TAG="step1_${M}x${N}x${K}"
echo "== ${M}x${N}x${K}, pass: $PASS =="

# Same memref-in / void-return shape as stage0_baseline.mlir, which is both
# what harness/bench_matmul.c's calling convention expects and what the EDSL
# frontend actually emits (to_tensor at the boundary, linalg.fill zeroing the
# accumulator, materialize_in_destination on the way out).
cat > "$OUT/$TAG.mlir" <<EOF
module {
  func.func @matmul_blocked(%arg0: memref<${M}x${K}xf32>,
                            %arg1: memref<${K}x${N}xf32>,
                            %arg2: memref<${M}x${N}xf32>) {
    %0 = bufferization.to_tensor %arg0 restrict : memref<${M}x${K}xf32> to tensor<${M}x${K}xf32>
    %1 = bufferization.to_tensor %arg1 restrict : memref<${K}x${N}xf32> to tensor<${K}x${N}xf32>
    %cst = arith.constant 0.000000e+00 : f32
    %2 = tensor.empty() : tensor<${M}x${N}xf32>
    %3 = linalg.fill ins(%cst : f32) outs(%2 : tensor<${M}x${N}xf32>) -> tensor<${M}x${N}xf32>
    %4 = linalg.matmul ins(%0, %1 : tensor<${M}x${K}xf32>, tensor<${K}x${N}xf32>)
                       outs(%3 : tensor<${M}x${N}xf32>) -> tensor<${M}x${N}xf32>
    bufferization.materialize_in_destination %4 in restrict writable %arg2
      : (tensor<${M}x${N}xf32>, memref<${M}x${N}xf32>) -> ()
    return
  }
}
EOF

echo "== block + lower =="
"$EDSL_OPT" "$OUT/$TAG.mlir" $PASS -o "$OUT/${TAG}_blocked.mlir"
if ! grep -q "mlir_edsl.blocked" "$OUT/${TAG}_blocked.mlir"; then
  echo "!! chooseStrategy rejected this shape/strategy — it stayed on the old path."
  echo "   (M%MC, MC%MR, N%NC, NC%NR, K%KC, KC%8 must all resolve; see MatmulStrategy.cpp)"
  exit 1
fi
"$EDSL_OPT" "$OUT/${TAG}_blocked.mlir" -cpu-pipeline -o "$OUT/${TAG}_llvm.mlir"

echo "== translate + optimize + codegen =="
"$MLIR_TRANSLATE" --mlir-to-llvmir "$OUT/${TAG}_llvm.mlir" -o "$OUT/$TAG.ll"
"$LLVM_OPT" -passes='default<O3>' -mcpu="$MCPU" "$OUT/$TAG.ll" -S -o "$OUT/$TAG.opt.ll"
"$LLVM_LLC" -O3 -mcpu="$MCPU" -relocation-model=pic "$OUT/$TAG.opt.ll" -o "$OUT/$TAG.opt.s"
"$LLVM_LLC" -O3 -mcpu="$MCPU" -relocation-model=pic -filetype=obj "$OUT/$TAG.opt.ll" \
  -o "$OUT/${TAG}_kernel.o"
# The blocked path tiles ic x jc into an scf.forall, which becomes
# omp.parallel, so the kernel references __kmpc_*. Link the same libomp the
# JIT loads (MLIRExecutor::initialize) rather than letting dlopen fail.
LIBOMP="${LIBOMP:-$(grep -m1 '^LIBOMP_PATH' "$REPO/build/CMakeCache.txt" | cut -d= -f2)}"
if [ -z "$LIBOMP" ] || [ ! -e "$LIBOMP" ]; then
  echo "!! libomp not found (set LIBOMP=/path/to/libomp.so)" >&2
  exit 1
fi
# -rpath so the kernel resolves libomp on its own; putting its directory on
# LD_LIBRARY_PATH instead can shadow the toolchain's own libstdc++.
clang -shared -fPIC "$OUT/${TAG}_kernel.o" -o "$OUT/lib${TAG}_kernel.so" \
  "$LIBOMP" -Wl,-rpath,"$(dirname "$LIBOMP")"

echo "== hot loop (densest FMA block) =="
python3 - "$OUT/$TAG.opt.s" <<'PY'
import re, sys
src = open(sys.argv[1]).read().splitlines()
best = None
for i, l in enumerate(src):
    if re.match(r'^\.LBB\d+_\d+:', l):
        j, body = i + 1, []
        while j < len(src) and not re.match(r'^\.LBB\d+_\d+:|^# %bb', src[j]):
            body.append(src[j]); j += 1
        n = sum('vfmadd' in x for x in body)
        if best is None or n > best[0]:
            best = (n, l, body)
n, lab, body = best
counts = {}
for l in body:
    m = re.match(r'\s+([a-z][a-z0-9]*)\s', l)
    if m:
        counts[m.group(1)] = counts.get(m.group(1), 0) + 1
print("  ", lab.strip())
for k, v in sorted(counts.items(), key=lambda kv: -kv[1]):
    print(f"   {v:3d}  {k}")
spills = sum(1 for l in body if 'Spill' in l or 'Reload' in l)
print(f"   spill/reload: {spills} (want 0)")
PY
echo "memcpy calls: $(grep -c 'callq.*memcpy' "$OUT/$TAG.opt.s" || true) (want 0)"

echo "== correctness + GFLOPS (naive reference check inside bench_matmul) =="
cc -O2 -o "$OUT/bench_matmul" "$DIR/../harness/bench_matmul.c" -ldl
taskset -c 0 "$OUT/bench_matmul" "$OUT/lib${TAG}_kernel.so" matmul_blocked "$M" "$N" "$K" 10

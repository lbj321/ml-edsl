#!/usr/bin/env bash
# Stage 6: multithreaded matmul (jc-parallel scf.forall -> OpenMP).
#
#   run_stage6.sh [M N K]        (default 1024 1024 1024)
#
# Generates the M x N x K input into out/, specializes stage6_transform.mlir's
# tile sizes to the shape (NC = N/8 -> one jc block per thread,
# MC = min(128, M), KC = min(256, K)), then:
#   transform -> bufferize -> hoist + dealloc (Stage 4, unchanged)
#   -> scalarize A~ reads -> forall -> scf.parallel -> omp.parallel
#   -> -canonicalize (inlines the alloca_scope convert-scf-to-openmp wraps the
#      body in; AllocaScopeInliner fires since the per-thread buffers are
#      heap allocs, not allocas -- replaces production's AllocaScopeCleanupPass)
#   -> LLVM lowering (+ convert-openmp-to-llvm) -> opt -O3 -> llc -> .so
#      linked against libomp.
# No remainder handling: fails fast unless the tiles divide the shape.
# Structural checks at the end fail the script (exit 1).
#
# Env overrides: LIBOMP_DIR (default /home/larsan/anaconda3/lib -- LIBOMP_PATH
# in build/CMakeCache.txt), NTHREADS_BLOCKS (jc blocks, default 8).
set -euo pipefail

MLIR_OPT="${MLIR_OPT:-mlir-opt}"
MLIR_TRANSLATE="${MLIR_TRANSLATE:-mlir-translate}"
LLVM_OPT="${LLVM_OPT:-opt}"
LLVM_LLC="${LLVM_LLC:-llc}"
LIBOMP_DIR="${LIBOMP_DIR:-/home/larsan/anaconda3/lib}"
JC_BLOCKS="${NTHREADS_BLOCKS:-8}"

M="${1:-1024}"
N="${2:-1024}"
K="${3:-1024}"

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="$DIR/../out"
mkdir -p "$OUT"
TAG="stage6_${M}x${N}x${K}"

NC=$((N / JC_BLOCKS))
MC=$(( M < 128 ? M : 128 ))
KC=$(( K < 256 ? K : 256 ))
MR=4
NR=16
for chk in "N % JC_BLOCKS" "NC % NR" "M % MC" "MC % MR" "K % KC" "KC % 8"; do
  if (( $chk != 0 )); then
    echo "shape ${M}x${N}x${K} fails fast-path guard: ($chk) != 0 (NC=$NC MC=$MC KC=$KC)" >&2
    exit 2
  fi
done
echo "== ${M}x${N}x${K}: MR=$MR NR=$NR MC=$MC NC=$NC KC=$KC, $JC_BLOCKS jc blocks =="

cat > "$OUT/${TAG}_input.mlir" << EOF
func.func @matmul(%A: memref<${M}x${K}xf32>, %B: memref<${K}x${N}xf32>, %C: memref<${M}x${N}xf32>) {
  %a = bufferization.to_tensor %A restrict : memref<${M}x${K}xf32> to tensor<${M}x${K}xf32>
  %b = bufferization.to_tensor %B restrict : memref<${K}x${N}xf32> to tensor<${K}x${N}xf32>
  %cst = arith.constant 0.000000e+00 : f32
  %init = tensor.empty() : tensor<${M}x${N}xf32>
  %filled = linalg.fill ins(%cst : f32) outs(%init : tensor<${M}x${N}xf32>) -> tensor<${M}x${N}xf32>
  %res = linalg.matmul ins(%a, %b : tensor<${M}x${K}xf32>, tensor<${K}x${N}xf32>) outs(%filled : tensor<${M}x${N}xf32>) -> tensor<${M}x${N}xf32>
  bufferization.materialize_in_destination %res in restrict writable %C : (tensor<${M}x${N}xf32>, memref<${M}x${N}xf32>) -> ()
  return
}
EOF

sed -e "s/tile_sizes \[0, 128, 0\]/tile_sizes [0, $NC, 0]/" \
    -e "s/tile_sizes \[0, 0, 256\]/tile_sizes [0, 0, $KC]/" \
    -e "s/tile_sizes \[128, 0, 0\]/tile_sizes [$MC, 0, 0]/" \
    "$DIR/stage6_transform.mlir" > "$OUT/${TAG}_transform.mlir"

echo "== transform =="
"$MLIR_OPT" "$OUT/${TAG}_input.mlir" \
  -transform-preload-library="transform-library-paths=$OUT/${TAG}_transform.mlir" \
  -transform-interpreter \
  -canonicalize -cse \
  -o "$OUT/${TAG}_vectorized.mlir" > "$OUT/${TAG}_transform_log.txt" 2>&1

echo "== bufferize, hoist, dealloc (Stage 4) =="
"$MLIR_OPT" "$OUT/${TAG}_vectorized.mlir" \
  -eliminate-empty-tensors \
  -one-shot-bufferize="bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map" \
  -canonicalize -cse \
  -buffer-hoisting -buffer-loop-hoisting \
  -ownership-based-buffer-deallocation \
  -canonicalize \
  -buffer-deallocation-simplification \
  -bufferization-lower-deallocations \
  -canonicalize -cse \
  -o "$OUT/${TAG}_dealloc.mlir"

echo "== scalarize A~ tile reads =="
"$MLIR_OPT" "$OUT/${TAG}_dealloc.mlir" \
  -test-scalar-vector-transfer-lowering=allow-multiple-uses \
  -canonicalize -cse \
  -o "$OUT/${TAG}_scalarized.mlir"

echo "== scf.forall -> omp.parallel =="
"$MLIR_OPT" "$OUT/${TAG}_scalarized.mlir" \
  -scf-forall-to-parallel \
  -convert-scf-to-openmp \
  -canonicalize \
  -o "$OUT/${TAG}_omp.mlir"

echo "== lower to LLVM dialect =="
"$MLIR_OPT" "$OUT/${TAG}_omp.mlir" \
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
  -convert-openmp-to-llvm \
  -convert-func-to-llvm \
  -reconcile-unrealized-casts \
  -o "$OUT/${TAG}_llvm.mlir"

echo "== translate + optimize + codegen + link libomp =="
"$MLIR_TRANSLATE" --mlir-to-llvmir "$OUT/${TAG}_llvm.mlir" -o "$OUT/${TAG}.ll"
"$LLVM_OPT" -passes='default<O3>' -mcpu=skylake "$OUT/${TAG}.ll" -S -o "$OUT/${TAG}.opt.ll"
"$LLVM_LLC" -O3 -mcpu=skylake -relocation-model=pic "$OUT/${TAG}.opt.ll" -o "$OUT/${TAG}.opt.s"
"$LLVM_LLC" -O3 -mcpu=skylake -relocation-model=pic -filetype=obj "$OUT/${TAG}.opt.ll" -o "$OUT/${TAG}_kernel.o"
clang -shared -fPIC "$OUT/${TAG}_kernel.o" -L"$LIBOMP_DIR" -Wl,-rpath,"$LIBOMP_DIR" -lomp \
  -o "$OUT/lib${TAG}_kernel.so"

echo "== checks =="
python3 - "$OUT/${TAG}_dealloc.mlir" "$OUT/${TAG}_omp.mlir" "$OUT/${TAG}.opt.s" \
  "$OUT/${TAG}_loop_body.s" "$MR" << 'PYEOF'
import re, sys
dealloc_path, omp_path, asm_path, body_path, mr = sys.argv[1:6]
mr = int(mr)
failures = []

def check(name, got, want):
    ok = got == want
    print(f"{'ok  ' if ok else 'FAIL'} {name}: {got} (want {want})")
    if not ok:
        failures.append(name)

# --- after bufferization + dealloc: per-thread buffers inside the forall ---
mlir = open(dealloc_path).read().splitlines()
forall_line = next((i for i, l in enumerate(mlir) if "scf.forall" in l), None)
check("scf.forall present", forall_line is not None, True)
allocs = [i for i, l in enumerate(mlir) if "memref.alloc(" in l or "memref.alloc()" in l]
deallocs = [i for i, l in enumerate(mlir) if "memref.dealloc" in l]
check("memref.alloc count", len(allocs), 2)
check("memref.dealloc count", len(deallocs), 2)
# Hoisting them above the forall would share one buffer across threads.
check("allocs hoisted out of the forall (race)",
      sum(i < (forall_line or 0) for i in allocs), 0)
copies = [l for l in mlir if "memref.copy" in l]
check("memref.copy", len(copies), 0)

# --- after scf -> openmp + canonicalize ---
omp = open(omp_path).read()
check("omp.parallel", omp.count("omp.parallel"), 1)
check("memref.alloca_scope left after canonicalize", omp.count("memref.alloca_scope"), 0)

# --- asm ---
asm = open(asm_path).read().splitlines()
calls = [m.group(1) for l in asm if (m := re.match(r"\s+(?:callq|jmp)\s+(\S+)@PLT", l))]
check("__kmpc_fork_call", calls.count("__kmpc_fork_call"), 1)
check("malloc calls", calls.count("malloc"), 2)
check("free calls", calls.count("free"), 2)
check("memcpy calls", calls.count("memcpy"), 0)
check("memrefCopy calls", calls.count("memrefCopy"), 0)

label = None
body = []
for i, line in enumerate(asm):
    m = re.match(r'(\.LBB\d+_\d+):', line)
    if m:
        label = m.group(1)
    if "vfmadd231ps" in line and label:
        start = next(j for j in range(i, -1, -1) if asm[j].startswith(label + ":"))
        end = next(j for j in range(i, len(asm)) if re.search(r'\bj\w+\s+' + re.escape(label) + r'\b', asm[j]))
        body = asm[start:end + 1]
        break
open(body_path, "w").write("\n".join(body) + "\n")
check("kernel vfmadd231ps", sum("vfmadd231ps" in l for l in body), 2 * mr)
check("kernel vbroadcastss", sum("vbroadcastss" in l for l in body), mr)
check("kernel B loads (vmovups/vmovaps)", sum(bool(re.search(r"vmov[au]ps", l)) for l in body), 2)
check("kernel spills/reloads", sum(("Spill" in l or "Reload" in l) for l in body), 0)

if failures:
    print(f"{len(failures)} check(s) FAILED: {', '.join(failures)}")
    sys.exit(1)
print("all checks passed")
PYEOF

if [ ! -x "$OUT/bench_matmul" ] || [ "$DIR/../harness/bench_matmul.c" -nt "$OUT/bench_matmul" ]; then
  clang -O2 -o "$OUT/bench_matmul" "$DIR/../harness/bench_matmul.c" -ldl
fi
echo "wrote $OUT/lib${TAG}_kernel.so (symbol: matmul)"
echo "run:  OMP_NUM_THREADS=8 OMP_PROC_BIND=close OMP_PLACES=cores taskset -c 0-7 \\"
echo "        $OUT/bench_matmul $OUT/lib${TAG}_kernel.so matmul $M $N $K"

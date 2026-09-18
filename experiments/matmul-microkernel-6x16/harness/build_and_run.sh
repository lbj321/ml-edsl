#!/usr/bin/env bash
# Stage 0 harness driver: lowers a .mlir matmul through the production
# `-cpu-pipeline` (or any other mlir-edsl-opt pipeline/transform script you
# pass via EXTRA_OPT_ARGS), links it into a shared library, and benchmarks
# it with bench_matmul.
#
# Usage:
#   ./build_and_run.sh <input.mlir> <symbol> M N K [repeats]
#
# Env overrides:
#   MLIR_EDSL_OPT   path to the mlir-edsl-opt binary
#                    (default: <repo>/build/cpp/tools/mlir-edsl-opt/mlir-edsl-opt)
#   LLVM_BIN        dir containing mlir-translate/llc (required, no default --
#                   point it at your local pinned LLVM build's bin/ dir)
#   LIBOMP_DIR      dir containing libomp.so for linking (required, no default)
#   EXTRA_OPT_ARGS  extra args passed to mlir-edsl-opt instead of -cpu-pipeline,
#                   e.g. a transform-interpreter invocation for later stages.
#
# Requires taskset (util-linux) on PATH to pin the benchmark to one core.
set -euo pipefail

if [ "$#" -lt 5 ]; then
  echo "usage: $0 <input.mlir> <symbol> M N K [repeats]" >&2
  exit 1
fi

INPUT_MLIR="$1"
SYMBOL="$2"
M="$3"
N="$4"
K="$5"
REPEATS="${6:-20}"

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$DIR/../../.." && pwd)"
OUT="$DIR/../out"
mkdir -p "$OUT"

MLIR_EDSL_OPT="${MLIR_EDSL_OPT:-$REPO_ROOT/build/cpp/tools/mlir-edsl-opt/mlir-edsl-opt}"
: "${LLVM_BIN:?set LLVM_BIN to the bin dir of your local LLVM build}"
: "${LIBOMP_DIR:?set LIBOMP_DIR to a dir containing libomp.so}"

BASENAME="$(basename "$INPUT_MLIR" .mlir)"
LOWERED="$OUT/${BASENAME}_lowered.mlir"
LL="$OUT/${BASENAME}.ll"
OBJ="$OUT/${BASENAME}.o"
SO="$OUT/lib${BASENAME}.so"

echo "== lowering =="
if [ -n "${EXTRA_OPT_ARGS:-}" ]; then
  # shellcheck disable=SC2086
  "$MLIR_EDSL_OPT" "$INPUT_MLIR" $EXTRA_OPT_ARGS -o "$LOWERED"
else
  "$MLIR_EDSL_OPT" "$INPUT_MLIR" -cpu-pipeline -o "$LOWERED"
fi

echo "== translate to LLVM IR =="
"$LLVM_BIN/mlir-translate" --mlir-to-llvmir "$LOWERED" -o "$LL"

echo "== llc (skylake, PIC) =="
"$LLVM_BIN/llc" -mcpu=skylake -O3 -relocation-model=pic -filetype=obj "$LL" -o "$OBJ"

echo "== link shared library =="
clang -shared -fPIC "$OBJ" -L"$LIBOMP_DIR" -Wl,-rpath,"$LIBOMP_DIR" -lomp -lm -o "$SO"

echo "== build harness (if needed) =="
if [ ! -x "$DIR/bench_matmul" ] || [ "$DIR/bench_matmul.c" -nt "$DIR/bench_matmul" ]; then
  clang -O2 -o "$DIR/bench_matmul" "$DIR/bench_matmul.c" -ldl
fi

echo "== run (pinned to core 0) =="
taskset -c 0 "$DIR/bench_matmul" "$SO" "$SYMBOL" "$M" "$N" "$K" "$REPEATS"

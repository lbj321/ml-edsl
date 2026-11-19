#!/usr/bin/env bash
# Investigates the Stage 2 GFLOPS gap vs Stage 1 by sweeping MC and NC
# independently to separate two hypotheses:
#
#   H1 (panel > L1 -> L2-bound): per-flop rate drops once the A panel
#      (MC*KC*4 bytes) plus the resident B tile (NR*KC*4 = 16KB) exceed L1d
#      (32KB), i.e. once MC exceeds roughly (32KB-16KB)/(KC*4) = ~16 columns
#      here. Once past that point, since the whole A panel (up to 172KB) and
#      B panel still fit L2 (256KB), the rate should plateau rather than keep
#      falling as MC grows further, and should NOT depend much on NC (jr
#      repeat count), since repeats re-hit L2, not DRAM.
#   H2 (re-streaming cost that grows with NC): if instead GFLOPS keeps
#      falling as NC grows at fixed (large) MC, the cost scales with how many
#      times the A panel gets re-swept, not just its size.
#
# Requires: MLIR_OPT, MLIR_TRANSLATE, LLVM_OPT, LLVM_LLC env vars (same as
# run_stage2.sh), and clang + taskset on PATH.
set -euo pipefail

MLIR_OPT="${MLIR_OPT:-mlir-opt}"
MLIR_TRANSLATE="${MLIR_TRANSLATE:-mlir-translate}"
LLVM_OPT="${LLVM_OPT:-opt}"
LLVM_LLC="${LLVM_LLC:-llc}"
KC=256

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="$DIR/../out/sweep"
mkdir -p "$OUT"

gen_and_run() {
  local MC="$1" NC="$2" TAG="mc${1}_nc${2}"
  local SYM="macrokernel_${TAG}"
  local SRC="$OUT/${TAG}.mlir"
  local XFORM="$OUT/${TAG}_transform.mlir"

  cat > "$SRC" << EOF
func.func @${SYM}(%A: memref<${KC}x${MC}xf32>, %B: memref<${KC}x${NC}xf32>, %C: memref<${MC}x${NC}xf32>) {
  %a = bufferization.to_tensor %A restrict : memref<${KC}x${MC}xf32> to tensor<${KC}x${MC}xf32>
  %b = bufferization.to_tensor %B restrict : memref<${KC}x${NC}xf32> to tensor<${KC}x${NC}xf32>
  %cst = arith.constant 0.000000e+00 : f32
  %init = tensor.empty() : tensor<${MC}x${NC}xf32>
  %filled = linalg.fill ins(%cst : f32) outs(%init : tensor<${MC}x${NC}xf32>) -> tensor<${MC}x${NC}xf32>
  %res = linalg.matmul_transpose_a ins(%a, %b : tensor<${KC}x${MC}xf32>, tensor<${KC}x${NC}xf32>) outs(%filled : tensor<${MC}x${NC}xf32>) -> tensor<${MC}x${NC}xf32>
  bufferization.materialize_in_destination %res in restrict writable %C : (tensor<${MC}x${NC}xf32>, memref<${MC}x${NC}xf32>) -> ()
  return
}
EOF

  cat > "$XFORM" << 'EOF'
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %mm = transform.structured.match ops{["linalg.matmul_transpose_a"]} in %root
      : (!transform.any_op) -> !transform.any_op
    %tiled_n, %jr = transform.structured.tile_using_for %mm tile_sizes [0, 16, 0]
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    %tiled_mn, %ir = transform.structured.tile_using_for %tiled_n tile_sizes [6, 0, 0]
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    %tiled_mnk, %kloop = transform.structured.tile_using_for %tiled_mn tile_sizes [0, 0, 1]
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    %f = transform.structured.match ops{["func.func"]} in %root
      : (!transform.any_op) -> !transform.any_op
    %fv = transform.structured.vectorize_children_and_apply_patterns %f
      : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %fv {
      transform.apply_patterns.vector.lower_contraction lowering_strategy = "outerproduct"
      transform.apply_patterns.vector.lower_outerproduct
      transform.apply_patterns.canonicalization
    } : !transform.any_op
    %forops = transform.structured.match ops{["scf.for"]} in %fv
      : (!transform.any_op) -> !transform.any_op
    transform.loop.hoist_loop_invariant_subsets %forops : !transform.any_op
    transform.yield
  }
}
EOF

  "$MLIR_OPT" "$SRC" \
    -transform-preload-library="transform-library-paths=$XFORM" \
    -transform-interpreter -canonicalize -cse \
    -o "$OUT/${TAG}_v.mlir"
  "$MLIR_OPT" "$OUT/${TAG}_v.mlir" \
    -eliminate-empty-tensors \
    -one-shot-bufferize="bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map" \
    -canonicalize -cse \
    -o "$OUT/${TAG}_b.mlir"
  "$MLIR_OPT" "$OUT/${TAG}_b.mlir" \
    -test-scalar-vector-transfer-lowering=allow-multiple-uses \
    -canonicalize -cse \
    -o "$OUT/${TAG}_s.mlir"
  "$MLIR_OPT" "$OUT/${TAG}_s.mlir" \
    -convert-vector-to-scf -buffer-hoisting -buffer-loop-hoisting -canonicalize \
    -convert-vector-to-llvm="enable-x86vector" -convert-ub-to-llvm -convert-scf-to-cf \
    -expand-strided-metadata -lower-affine -convert-arith-to-llvm \
    -finalize-memref-to-llvm -convert-cf-to-llvm -convert-func-to-llvm \
    -reconcile-unrealized-casts \
    -o "$OUT/${TAG}_llvm.mlir"
  "$MLIR_TRANSLATE" --mlir-to-llvmir "$OUT/${TAG}_llvm.mlir" -o "$OUT/${TAG}.ll"
  "$LLVM_OPT" -passes='default<O3>' -mcpu=skylake "$OUT/${TAG}.ll" -S -o "$OUT/${TAG}.opt.ll"
  "$LLVM_LLC" -mcpu=skylake -O3 -relocation-model=pic -filetype=obj "$OUT/${TAG}.opt.ll" -o "$OUT/${TAG}.o"
  clang -shared -fPIC "$OUT/${TAG}.o" -o "$OUT/lib${TAG}.so"

  taskset -c 0 "$DIR/../harness/bench_macrokernel" "$OUT/lib${TAG}.so" "$SYM" "$MC" "$NC" "$KC" 30 \
    | sed -n '2p' | sed "s/^/MC=$MC NC=$NC: /"
}

clang -O2 -o "$DIR/../harness/bench_macrokernel" "$DIR/../harness/bench_macrokernel.c" -ldl

echo "=== sweep 1: vary MC, NC fixed at 16 (single jr iteration -- tests H1) ==="
for MC in 6 12 18 24 36 48 96 168; do
  gen_and_run "$MC" 16
done

echo "=== sweep 2: vary NC, MC fixed at 168 (tests H2: does repeat count matter independently of size) ==="
for NC in 16 32 64; do
  gen_and_run 168 "$NC"
done

// Transform-dialect library: pack the matmul directly via
// transform.structured.pack (the same building block -linalg-block-pack-matmul
// uses internally) instead of shelling out to that pass, so packing is a
// scriptable stage in its own right that can later be driven off an outer
// tile handle instead of a fixed matrix-wide size.
//
// Run in isolation as its own --transform-interpreter pass over matmul.mlir
// so the packed IR can be inspected on its own before any tiling/
// vectorization is layered on top.
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%module: !transform.any_op {transform.readonly}) {
    // Pack A, B, and the fill-init accumulator into 32x32 blocks. Expect:
    // linalg.pack for A/B, a fresh linalg.fill re-materialized directly at
    // packed shape (the zero-accumulator case is special-cased - no
    // pack/unpack needed for it), and a 6-loop linalg.generic replacing the
    // matmul, plus a trailing linalg.unpack back to the original shape.
    %matmul = transform.structured.match ops{["linalg.matmul"]} in %module
        : (!transform.any_op) -> !transform.any_op
    %packed = transform.structured.pack %matmul packed_sizes = [32, 32, 32]
        : (!transform.any_op) -> !transform.any_op

    %func = transform.structured.match ops{["func.func"]} attributes{sym_name = "matmul"} in %module
        : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func {
      transform.apply_patterns.canonicalization
    } : !transform.any_op
    transform.apply_cse to %func : !transform.any_op

    transform.yield
  }
}

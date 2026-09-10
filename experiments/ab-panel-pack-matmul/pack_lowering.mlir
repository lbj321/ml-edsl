// Transform-dialect library: lower linalg.pack/linalg.unpack into their
// constituent ops (tensor.pad+expand_shape+transpose /
// empty+transpose+collapse_shape+extract_slice). Run directly on
// out/packed.mlir (pre-tiling) to answer one question early: does
// -eliminate-empty-tensors see through the unpack's lowering here, now that
// -linalg-block-pack-matmul's unpack already targets the real destination
// (linalg.fill's result, tracing back to %arg2) instead of a fresh
// tensor.empty the way our hand-rolled transform.structured.pack + manual
// lower_unpack did in ../matmul-per-tile-packing?
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%module: !transform.any_op {transform.readonly}) {
    %packs = transform.structured.match ops{["linalg.pack"]} in %module
        : (!transform.any_op) -> !transform.op<"linalg.pack">
    transform.structured.lower_pack %packs
        : (!transform.op<"linalg.pack">)
        -> (!transform.op<"tensor.pad">, !transform.op<"tensor.expand_shape">, !transform.op<"linalg.transpose">)

    %unpacks = transform.structured.match ops{["linalg.unpack"]} in %module
        : (!transform.any_op) -> !transform.op<"linalg.unpack">
    transform.structured.lower_unpack %unpacks
        : (!transform.op<"linalg.unpack">)
        -> (!transform.op<"tensor.empty">, !transform.op<"linalg.transpose">,
            !transform.op<"tensor.collapse_shape">, !transform.op<"tensor.extract_slice">)

    transform.yield
  }
}

// Transform-dialect library: lower linalg.pack/linalg.unpack into their
// constituent ops (tensor.pad+expand_shape+transpose / empty+transpose+
// collapse_shape+extract_slice). Runs as its own --transform-interpreter
// pass over out/vectorized.mlir, before bufferization.
//
// lower_pack/lower_unpack need concrete !transform.op<"linalg.pack">/
// <"linalg.unpack"> handle types, not !transform.any_op, or they reject
// the match.
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

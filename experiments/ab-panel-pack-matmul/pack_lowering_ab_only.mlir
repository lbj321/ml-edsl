// Transform-dialect library: lower only linalg.pack (A/B) into its
// constituent ops, leaving linalg.unpack (C) untouched so
// --linalg-lower-unpack-direct (LowerUnpackDirectPass.cpp) can be tested on
// it afterward instead of the stock transform.structured.lower_unpack.
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%module: !transform.any_op {transform.readonly}) {
    %packs = transform.structured.match ops{["linalg.pack"]} in %module
        : (!transform.any_op) -> !transform.op<"linalg.pack">
    transform.structured.lower_pack %packs
        : (!transform.op<"linalg.pack">)
        -> (!transform.op<"tensor.pad">, !transform.op<"tensor.expand_shape">, !transform.op<"linalg.transpose">)

    transform.yield
  }
}

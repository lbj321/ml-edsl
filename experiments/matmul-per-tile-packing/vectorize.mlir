// Vectorize the tiled 8x8x8 compute generic into a real FMA-lowerable
// vector.contract, then lower it to vector.outerproduct pre-bufferize
// (matching cpp/src/MLIRLoweringPasses.cpp's VectorContractToOuterProductPass
// ordering exactly).
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%module: !transform.any_op {transform.readonly}) {
    // linalg::vectorize always lowers a reduction to arith.mulf +
    // vector.multi_reduction with BROADCAST-shaped operands, never
    // vector.contract directly - see cpp/src/MLIRLoweringPasses.cpp's
    // LinalgMatmulToContractPass comment for why the real compiler avoids
    // this path via its own C++ pass instead.
    %generic = transform.structured.match ops{["linalg.generic"]} in %module
        : (!transform.any_op) -> !transform.any_op
    transform.structured.vectorize %generic : !transform.any_op

    %func = transform.structured.match ops{["func.func"]} attributes{sym_name = "matmul"} in %module
        : (!transform.any_op) -> !transform.any_op

    // transfer_permutation_patterns strips the broadcast dim from each
    // transfer_read into an explicit vector.broadcast; reduction_to_contract
    // then folds mulf+multi_reduction into vector.contract AND reduces its
    // rank using that explicit broadcast - both needed together, in the
    // same apply_patterns block, or the contract keeps its broadcast shape
    // and outerproduct lowering silently falls back to a scalar expansion.
    transform.apply_patterns to %func {
      transform.apply_patterns.vector.transfer_permutation_patterns
      transform.apply_patterns.vector.reduction_to_contract
    } : !transform.any_op
    transform.apply_cse to %func : !transform.any_op

    transform.apply_patterns to %func {
      transform.apply_patterns.vector.lower_contraction
    } : !transform.any_op
    transform.apply_cse to %func : !transform.any_op

    transform.yield
  }
}

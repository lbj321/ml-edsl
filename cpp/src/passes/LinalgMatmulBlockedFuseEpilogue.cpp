//===- LinalgMatmulBlockedFuseEpilogue.cpp - epilogue into the forall -----===//
//
// linalg-matmul-blocked-fuse-epilogue: runs between the distribute and tile
// passes (see BlockedStage in MatmulStrategy.h). Pulls the elementwise
// linalg.generic chain consuming a distributed matmul (bias add, relu, ...)
// into its ic x jc scf.forall, so each MC x NC block of C is finished while
// it is still in cache instead of in extra sweeps over the whole of C.
//
//===----------------------------------------------------------------------===//

#include "mlir_edsl/MLIRLoweringPasses.h"
#include "mlir_edsl/MatmulStrategy.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"

namespace {

// The linalg.generic to fuse next, or null when the chain ends. Only a
// single-result all-parallel generic reading the forall result as an input is
// safe to tile along the forall's dimensions; anything else (the padded
// case's linalg.copy, a reduction, a second user) is left where it is.
static mlir::linalg::GenericOp nextEpilogueOp(mlir::OpResult forallResult) {
  if (!forallResult.hasOneUse())
    return nullptr;
  mlir::OpOperand &use = *forallResult.getUses().begin();
  auto generic = llvm::dyn_cast<mlir::linalg::GenericOp>(use.getOwner());
  if (!generic || generic.getNumResults() != 1)
    return nullptr;
  if (generic.getNumReductionLoops() != 0)
    return nullptr;
  if (!generic.isDpsInput(&use))
    return nullptr;
  return generic;
}

// The parallel_insert_slice yielding `tiledResult` out of its forall.
static mlir::tensor::ParallelInsertSliceOp
yieldingSlice(mlir::Value tiledResult) {
  if (!tiledResult.hasOneUse())
    return nullptr;
  return llvm::dyn_cast<mlir::tensor::ParallelInsertSliceOp>(
      *tiledResult.user_begin());
}

// The fused generic's init becomes a new shared_out of the forall, so it must
// dominate the forall. A tensor.empty built after the chain's producer (as the
// builders do for each op's result) does not, and has no operands to stop it
// moving up.
static void hoistEmptyInits(mlir::IRRewriter &rewriter,
                            mlir::linalg::GenericOp generic,
                            mlir::scf::ForallOp forall) {
  for (mlir::Value init : generic.getDpsInits()) {
    auto empty = init.getDefiningOp<mlir::tensor::EmptyOp>();
    if (empty && empty->getBlock() == forall->getBlock() &&
        forall->isBeforeInBlock(empty))
      rewriter.moveOpBefore(empty, forall);
  }
}

// Fuses the generic chain after `tile`'s forall, returning how many generics
// were fused. Each fusion rebuilds the forall with one more result, so the
// loop re-derives everything from `loops` and the new slice.
static unsigned fuseEpilogueChain(mlir::IRRewriter &rewriter,
                                  mlir::linalg::MatmulOp tile) {
  auto forall = llvm::dyn_cast<mlir::scf::ForallOp>(tile->getParentOp());
  auto slice = yieldingSlice(tile->getResult(0));
  if (!forall || !slice)
    return 0;

  llvm::SmallVector<mlir::LoopLikeOpInterface> loops = {forall};
  unsigned fused = 0;
  while (true) {
    auto dest = llvm::dyn_cast<mlir::BlockArgument>(slice.getDest());
    if (!dest)
      break;
    mlir::OpResult result =
        forall.getTiedOpResult(forall.getTiedOpOperand(dest));
    mlir::linalg::GenericOp next = nextEpilogueOp(result);
    if (!next)
      break;
    hoistEmptyInits(rewriter, next, forall);

    rewriter.setInsertionPoint(slice);
    auto fusion =
        mlir::scf::tileAndFuseConsumerOfSlice(rewriter, slice, loops);
    if (mlir::failed(fusion))
      break;
    ++fused;

    forall = llvm::cast<mlir::scf::ForallOp>(loops.front().getOperation());
    slice = yieldingSlice(fusion->tiledOps.front()->getResult(0));
    if (!slice)
      break;
  }
  return fused;
}

struct LinalgMatmulBlockedFuseEpiloguePass
    : public mlir::PassWrapper<LinalgMatmulBlockedFuseEpiloguePass,
                               mlir::OperationPass<mlir::func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      LinalgMatmulBlockedFuseEpiloguePass)

  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::scf::SCFDialect, mlir::tensor::TensorDialect,
                    mlir::linalg::LinalgDialect>();
  }

  llvm::StringRef getArgument() const override {
    return "linalg-matmul-blocked-fuse-epilogue";
  }
  llvm::StringRef getDescription() const override {
    return "Fuse the elementwise epilogue of a distributed blocked matmul "
           "into its ic x jc scf.forall";
  }

  void runOnOperation() override {
    mlir::func::FuncOp func = getOperation();
    mlir::IRRewriter rewriter(func->getContext());

    llvm::SmallVector<
        std::pair<mlir::linalg::MatmulOp, mlir_edsl::MatmulStrategy>>
        tiles;
    if (mlir::failed(mlir_edsl::collectBlockedTilesAtStage(
            func, mlir_edsl::BlockedStage::Distributed, tiles)))
      return signalPassFailure();

    // Fusion is an optimization: a chain that cannot be fused stays as
    // separate ops and still computes the right result.
    for (auto &[tile, strategy] : tiles)
      (void)fuseEpilogueChain(rewriter, tile);
  }
};

} // namespace

namespace mlir_edsl {

std::unique_ptr<mlir::Pass> createLinalgMatmulBlockedFuseEpiloguePass() {
  return std::make_unique<LinalgMatmulBlockedFuseEpiloguePass>();
}

} // namespace mlir_edsl

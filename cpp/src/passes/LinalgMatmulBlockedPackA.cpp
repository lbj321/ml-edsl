//===- LinalgMatmulBlockedPackA.cpp - pack A into k-major panels ----------===//
//
// linalg-matmul-blocked-pack-a: the last of the six blocked matmul passes
// (see BlockedStage in MatmulStrategy.h). Hoists the nofold pad pack-prepare
// put on each Padded tile's A out of the loops marked with
// kBlockedHoistAttrName, with a transpose, which packs A into k-major panels.
// Then removes the markers and leaves the tile at stage Packed.
//
//===----------------------------------------------------------------------===//

#include "TilingUtils.h"

#include "mlir_edsl/MLIRLoweringPasses.h"
#include "mlir_edsl/MatmulStrategy.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"

namespace {

using mlir_edsl::applyPatternSetTo;
using mlir_edsl::BlockedStage;

// A~ = (MC/MR) x KC x 1 x MR: hoists A's pad out of the marked loops with a
// [1, 0] transpose, so each k step reads a contiguous MR-element row instead
// of MR scalars from MR different rows. The un-transpose this leaves in front
// of the tile is already inside the k-loop, so it only transposes that row.
static mlir::LogicalResult packA(mlir::IRRewriter &rewriter,
                                 mlir::tensor::PadOp padA,
                                 int64_t hoistDepth) {
  mlir::tensor::PadOp hoistedPad;
  llvm::SmallVector<mlir::linalg::TransposeOp> transposeOps;
  auto packed = mlir::linalg::hoistPaddingOnTensors(
      rewriter, padA, hoistDepth, /*transposeVector=*/{1, 0}, hoistedPad,
      transposeOps);
  if (mlir::failed(packed))
    return mlir::failure();
  rewriter.replaceOp(padA, *packed);

  // transposeOps[0] is the packing transpose, [1] the un-transpose put back
  // in front of the tile.
  if (transposeOps.size() != 2)
    return mlir::failure();

  // A's pad is zero-width — the transpose is the packing copy — and
  // tensor.pad is not destination-style, so it cannot be fused with the
  // fusion utilities. Decomposing it lets the transpose read A directly.
  mlir::RewritePatternSet patterns(rewriter.getContext());
  mlir::linalg::populateDecomposePadPatterns(patterns);
  // LinalgVectorizationPass vectorizes the packing transpose later.
  return applyPatternSetTo({hoistedPad.getOperation()}, std::move(patterns));
}

// Packs the tile's A if pack-prepare padded it, then ends the marker
// hand-over: the markers go no further than this pass.
static mlir::LogicalResult finishPacking(mlir::IRRewriter &rewriter,
                                         mlir::Operation *tile) {
  llvm::SmallVector<mlir::scf::ForOp> hoistLoops =
      mlir_edsl::collectHoistLoops(tile);
  mlir::tensor::PadOp padA = mlir_edsl::nofoldPadOperand(tile, 0);
  if (padA && mlir::failed(packA(rewriter, padA, hoistLoops.size())))
    return mlir::failure();

  for (mlir::scf::ForOp loop : hoistLoops)
    loop->removeAttr(mlir_edsl::kBlockedHoistAttrName);
  mlir_edsl::setBlockedStage(tile, BlockedStage::Packed);
  return mlir::success();
}

struct LinalgMatmulBlockedPackAPass
    : public mlir::PassWrapper<LinalgMatmulBlockedPackAPass,
                               mlir::OperationPass<mlir::func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LinalgMatmulBlockedPackAPass)

  // Hoisting builds scf.for packing loops, tensor.empty and linalg.transpose.
  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::scf::SCFDialect, mlir::tensor::TensorDialect,
                    mlir::linalg::LinalgDialect, mlir::arith::ArithDialect>();
  }

  llvm::StringRef getArgument() const override {
    return "linalg-matmul-blocked-pack-a";
  }
  llvm::StringRef getDescription() const override {
    return "Pack the A operand of padded blocked matmuls into k-major panels "
           "hoisted out of the k and register loops";
  }

  void runOnOperation() override {
    mlir::func::FuncOp func = getOperation();
    mlir::IRRewriter rewriter(func->getContext());

    llvm::SmallVector<
        std::pair<mlir::linalg::MatmulOp, mlir_edsl::MatmulStrategy>>
        tiles;
    if (mlir::failed(mlir_edsl::collectBlockedTilesAtStage(
            func, BlockedStage::Padded, tiles)))
      return signalPassFailure();

    // A blocked tile cannot fall back to the older passes, which skip it.
    for (auto &[tile, strategy] : tiles) {
      if (mlir::failed(finishPacking(rewriter, tile))) {
        func->emitError("linalg-matmul-blocked-pack-a failed");
        return signalPassFailure();
      }
    }
  }
};

} // namespace

namespace mlir_edsl {

std::unique_ptr<mlir::Pass> createLinalgMatmulBlockedPackAPass() {
  return std::make_unique<LinalgMatmulBlockedPackAPass>();
}

} // namespace mlir_edsl

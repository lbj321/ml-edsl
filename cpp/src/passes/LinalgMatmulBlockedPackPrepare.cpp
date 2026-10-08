//===- LinalgMatmulBlockedPackPrepare.cpp - pad the operands to pack ------===//
//
// linalg-matmul-blocked-pack-prepare: the fourth of the six blocked matmul
// passes (see BlockedStage in MatmulStrategy.h). Decides which operands of
// each Kernel MR x NR x 1 tile are worth packing, puts a nofold tensor.pad on
// each of them, and leaves the tile at stage Padded. That pad is the only
// record of the decision pack-b and pack-a read.
//
//===----------------------------------------------------------------------===//

#include "TilingUtils.h"

#include "mlir_edsl/MLIRLoweringPasses.h"
#include "mlir_edsl/MatmulStrategy.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Transforms/Transforms.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"

namespace {

using mlir_edsl::applyPatternSetTo;
using mlir_edsl::BlockedStage;

// Collapses the slice chains feeding the tile's A and B into single slices of
// the original tensors. Every tiling level slices the previous level's slice,
// so the tile's operands are defined *inside* the loops we want to hoist out
// of, and hoistPaddingOnTensors refuses with "Source not defined outside of
// loops". The chains are followed back from the operands rather than found
// under the forall, which canonicalize may have removed. Seeding the whole
// chain also lets the driver erase the outer slices once they are dead.
static mlir::LogicalResult mergeSliceChains(mlir::Operation *tile) {
  llvm::SmallVector<mlir::Operation *> slices;
  for (mlir::Value operand : tile->getOperands().take_front(2))
    for (auto slice = operand.getDefiningOp<mlir::tensor::ExtractSliceOp>();
         slice;
         slice = slice.getSource().getDefiningOp<mlir::tensor::ExtractSliceOp>())
      slices.push_back(slice);
  mlir::RewritePatternSet patterns(tile->getContext());
  mlir::tensor::populateMergeConsecutiveInsertExtractSlicePatterns(patterns);
  return applyPatternSetTo(slices, std::move(patterns));
}

// Pads every iteration dimension of the tile. The slices are already exactly
// MR x NR x 1 so no element is actually added; the nofold flag is what forces
// the pad to survive as a real copy, which *is* the packing. C is never padded
// — it is accumulated in place across k and pc — which is also what keeps the
// marked loops intact: padding it would make hoistPaddingOnTensors rebuild
// them to carry the pad through iter_args.
static mlir::LogicalResult padTile(mlir::IRRewriter &rewriter,
                                   mlir::Operation *&tile, bool packA,
                                   bool packB) {
  auto linalgTile = llvm::dyn_cast<mlir::linalg::LinalgOp>(tile);
  if (!linalgTile)
    return mlir::failure();

  mlir::Attribute zero = rewriter.getZeroAttr(rewriter.getF32Type());
  mlir::linalg::LinalgPaddingOptions padOpts;
  padOpts.setPaddingValues({zero, zero, zero});
  padOpts.setPaddingDimensions({0, 1, 2});
  padOpts.setNofoldFlags({packA, packB, false});
  padOpts.setCopyBackOp(mlir::linalg::LinalgPaddingOptions::CopyBackOp::None);

  mlir::linalg::LinalgOp paddedOp;
  llvm::SmallVector<mlir::Value> replacements;
  llvm::SmallVector<mlir::tensor::PadOp> padOps;
  rewriter.setInsertionPoint(tile);
  if (mlir::failed(mlir::linalg::rewriteAsPaddedOp(
          rewriter, linalgTile, padOpts, paddedOp, replacements, padOps)))
    return mlir::failure();
  // rewriteAsPaddedOp clones the op onto the padded operands (carrying its
  // attributes, so the marker survives) but leaves the original in place for
  // the caller to replace.
  rewriter.replaceOp(tile, replacements);
  tile = paddedOp.getOperation();
  return mlir::success();
}

// Whether the operand's MR x KC (A) or KC x NR (B) block is its whole
// original tensor, read through the slice mergeSliceChains left. Such an
// operand has nothing to gather: B's rows, for one, are then already the
// contiguous NR-wide rows of B~.
static bool isWholeTensorOperand(mlir::Operation *tile, unsigned operand,
                                 llvm::ArrayRef<int64_t> blockShape) {
  auto slice =
      tile->getOperand(operand).getDefiningOp<mlir::tensor::ExtractSliceOp>();
  return !slice || slice.getSourceType().getShape() == blockShape;
}

// Pads the operands one kernel tile will pack. Packing is skipped where it
// cannot pay: with no register loop (ir, jr) left to hoist out of, each panel
// would be used by a single microtile; and an operand whose block is the
// whole original tensor has nothing to gather. The tile still advances, so
// pack-b and pack-a see every tile and pack-a removes its markers.
static mlir::LogicalResult prepare(mlir::IRRewriter &rewriter,
                                   mlir::Operation *tile,
                                   const mlir_edsl::MatmulStrategy &s) {
  // collectHoistLoops' first loop is k.
  const bool hasRegisterLoop = mlir_edsl::collectHoistLoops(tile).size() > 1;
  if (hasRegisterLoop && (s.packA || s.packB)) {
    if (mlir::failed(mergeSliceChains(tile)))
      return mlir::failure();
    const bool withA = s.packA && !isWholeTensorOperand(tile, 0, {s.mr, s.kc});
    const bool withB = s.packB && !isWholeTensorOperand(tile, 1, {s.kc, s.nr});
    if ((withA || withB) && mlir::failed(padTile(rewriter, tile, withA, withB)))
      return mlir::failure();
  }

  mlir_edsl::setBlockedStage(tile, BlockedStage::Padded);
  return mlir::success();
}

struct LinalgMatmulBlockedPackPreparePass
    : public mlir::PassWrapper<LinalgMatmulBlockedPackPreparePass,
                               mlir::OperationPass<mlir::func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      LinalgMatmulBlockedPackPreparePass)

  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::tensor::TensorDialect, mlir::linalg::LinalgDialect,
                    mlir::arith::ArithDialect>();
  }

  llvm::StringRef getArgument() const override {
    return "linalg-matmul-blocked-pack-prepare";
  }
  llvm::StringRef getDescription() const override {
    return "Put nofold pads on the A and B operands of k-tiled blocked "
           "matmuls that are worth packing";
  }

  void runOnOperation() override {
    mlir::func::FuncOp func = getOperation();
    mlir::IRRewriter rewriter(func->getContext());

    llvm::SmallVector<
        std::pair<mlir::linalg::MatmulOp, mlir_edsl::MatmulStrategy>>
        tiles;
    if (mlir::failed(mlir_edsl::collectBlockedTilesAtStage(
            func, BlockedStage::Kernel, tiles)))
      return signalPassFailure();

    // A blocked tile cannot fall back to the older passes, which skip it.
    for (auto &[tile, strategy] : tiles) {
      if (mlir::failed(prepare(rewriter, tile, strategy))) {
        func->emitError("linalg-matmul-blocked-pack-prepare failed");
        return signalPassFailure();
      }
    }
  }
};

} // namespace

namespace mlir_edsl {

std::unique_ptr<mlir::Pass> createLinalgMatmulBlockedPackPreparePass() {
  return std::make_unique<LinalgMatmulBlockedPackPreparePass>();
}

} // namespace mlir_edsl

//===- LinalgMatmulBlockedPack.cpp - pack A and B into panels -------------===//
//
// linalg-matmul-blocked-pack: the last of the four blocked matmul passes
// (see BlockedStage in MatmulStrategy.h). Packs the A and B operands of each
// Kernel MR x NR x 1 tile into contiguous panels, hoisted out of the loops
// the tile and kernel passes marked with kBlockedHoistAttrName, and leaves the
// tile at stage Packed.
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
#include "mlir/Dialect/Tensor/Transforms/Transforms.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace {

using mlir_edsl::applyPatternSetTo;
using mlir_edsl::BlockedStage;

// The marked loops directly around `tile`, innermost first: k, then whichever
// of ir and jr survived. Canonicalize removes the ones with a single
// iteration, and their markers with them; k always has KC > 1 iterations.
static llvm::SmallVector<mlir::scf::ForOp>
collectHoistLoops(mlir::Operation *tile) {
  llvm::SmallVector<mlir::scf::ForOp> loops;
  for (auto loop = llvm::dyn_cast<mlir::scf::ForOp>(tile->getParentOp());
       loop && loop->hasAttr(mlir_edsl::kBlockedHoistAttrName);
       loop = llvm::dyn_cast<mlir::scf::ForOp>(loop->getParentOp()))
    loops.push_back(loop);
  return loops;
}

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

// B~ = (NC/NR) x KC x 1 x NR: hoists B's pad out of the marked loops with no
// transpose. The pad is a plain row copy, so the packed panel keeps B's (k, n)
// order and the microkernel reads contiguous NR-wide rows instead of striding
// by N.
static mlir::LogicalResult packB(mlir::IRRewriter &rewriter,
                                 mlir::Operation *tile, int64_t hoistDepth,
                                 const mlir_edsl::MatmulStrategy &s) {
  auto padB = tile->getOperand(1).getDefiningOp<mlir::tensor::PadOp>();
  if (!padB)
    return mlir::failure();

  mlir::tensor::PadOp hoistedPad;
  llvm::SmallVector<mlir::linalg::TransposeOp> transposeOps;
  auto packed = mlir::linalg::hoistPaddingOnTensors(
      rewriter, padB, hoistDepth, /*transposeVector=*/{}, hoistedPad,
      transposeOps);
  if (mlir::failed(packed))
    return mlir::failure();
  rewriter.replaceOp(padB, *packed);

  // The hoisted pad is one 1 x NR row, copied inside the k packing loop.
  // tensor.pad needs explicit vector sizes; it fails to vectorize without.
  return mlir_edsl::vectorizeCopyTile(rewriter, hoistedPad.getOperation(),
                                      hoistedPad->getParentOp(),
                                      /*vectorSizes=*/{1, s.nr});
}

// A~ = (MC/MR) x KC x 1 x MR: hoists A's pad out of the marked loops with a
// [1, 0] transpose, so each k step reads a contiguous MR-element row instead
// of MR scalars from MR different rows. The un-transpose this leaves in front
// of the tile is already inside the k-loop, so it only transposes that row.
static mlir::LogicalResult packA(mlir::IRRewriter &rewriter,
                                 mlir::Operation *tile, int64_t hoistDepth) {
  auto padA = tile->getOperand(0).getDefiningOp<mlir::tensor::PadOp>();
  if (!padA)
    return mlir::failure();

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

// Packs one kernel tile's operands. Packing is skipped where it cannot pay:
// with no register loop (ir, jr) left to hoist out of, each panel would be
// used by a single microtile; and an operand whose block is the whole
// original tensor has nothing to gather.
static mlir::LogicalResult pack(mlir::IRRewriter &rewriter,
                                mlir::Operation *tile,
                                const mlir_edsl::MatmulStrategy &s) {
  llvm::SmallVector<mlir::scf::ForOp> hoistLoops = collectHoistLoops(tile);
  const int64_t hoistDepth = hoistLoops.size();
  for (mlir::scf::ForOp loop : hoistLoops)
    loop->removeAttr(mlir_edsl::kBlockedHoistAttrName);

  const bool hasRegisterLoop = hoistDepth > 1; // hoistLoops[0] is k
  if (hasRegisterLoop && (s.packA || s.packB)) {
    if (mlir::failed(mergeSliceChains(tile)))
      return mlir::failure();
    const bool withA = s.packA && !isWholeTensorOperand(tile, 0, {s.mr, s.kc});
    const bool withB = s.packB && !isWholeTensorOperand(tile, 1, {s.kc, s.nr});
    if ((withA || withB) && mlir::failed(padTile(rewriter, tile, withA, withB)))
      return mlir::failure();
    if (withB && mlir::failed(packB(rewriter, tile, hoistDepth, s)))
      return mlir::failure();
    if (withA && mlir::failed(packA(rewriter, tile, hoistDepth)))
      return mlir::failure();
  }

  mlir_edsl::setBlockedStage(tile, BlockedStage::Packed);
  return mlir::success();
}

struct LinalgMatmulBlockedPackPass
    : public mlir::PassWrapper<LinalgMatmulBlockedPackPass,
                               mlir::OperationPass<mlir::func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LinalgMatmulBlockedPackPass)

  // Packing builds tensor.pad/empty, linalg.transpose and vector transfers.
  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::scf::SCFDialect, mlir::tensor::TensorDialect,
                    mlir::linalg::LinalgDialect, mlir::vector::VectorDialect,
                    mlir::arith::ArithDialect>();
  }

  llvm::StringRef getArgument() const override {
    return "linalg-matmul-blocked-pack";
  }
  llvm::StringRef getDescription() const override {
    return "Pack the A and B operands of k-tiled blocked matmuls into "
           "contiguous panels hoisted out of the k and register loops";
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
      if (mlir::failed(pack(rewriter, tile, strategy))) {
        func->emitError("linalg-matmul-blocked-pack failed");
        return signalPassFailure();
      }
    }
  }
};

} // namespace

namespace mlir_edsl {

std::unique_ptr<mlir::Pass> createLinalgMatmulBlockedPackPass() {
  return std::make_unique<LinalgMatmulBlockedPackPass>();
}

} // namespace mlir_edsl

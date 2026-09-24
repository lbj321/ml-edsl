//===- LinalgMatmulBlockedPack.cpp - pack A and B into panels -------------===//
//
// linalg-matmul-blocked-pack: the third of the four blocked matmul passes
// (see BlockedStage in MatmulStrategy.h). Packs the A and B operands of each
// Tiled MR x NR x KC tile into contiguous panels, hoisted out of the register
// loops the tile pass marked with kBlockedHoistAttrName, and leaves the tile
// at stage Packed.
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

using mlir_edsl::BlockedStage;
using mlir_edsl::tileOneLevel;

// Rows of B packed per iteration of the B~ packing loop. 8 is one f32 ymm
// register: each iteration copies an 8 x NR block with two vector transfers.
constexpr int64_t kPackTileRows = 8;

// Runs one pattern set greedily over `ops` only, plus the ops the patterns
// create. Scoping is what lets the caller keep holding handles: ops outside
// the set are never folded, rewritten or erased as dead. Kept deliberately
// narrow — no canonicalizer — because canonicalization at the wrong moment
// undoes two of the rewrites below (see packB).
static mlir::LogicalResult
applyPatternSetTo(llvm::ArrayRef<mlir::Operation *> ops,
                  mlir::RewritePatternSet &&patterns) {
  mlir::GreedyRewriteConfig config;
  config.setStrictness(mlir::GreedyRewriteStrictness::ExistingAndNewOps);
  return mlir::applyOpPatternsGreedily(ops, std::move(patterns), config);
}

// The marked register loops directly around `tile`, innermost first. The tile
// pass marks jr and ir; canonicalize removes the ones with a single
// iteration, and their markers with them.
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
// MR x NR x KC so no element is actually added; the nofold flag is what forces
// the pad to survive as a real copy, which *is* the packing. C is never padded
// — it is accumulated in place across pc — which is also what keeps the
// register loops intact: padding it would make hoistPaddingOnTensors rebuild
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

// B~ = (NC/NR) x KC x NR: hoists B's pad out of the register loops with no
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

  // Tile the packing copy to 8 rows and vectorize each 8 x NR block.
  // Untiled, it vectorizes to one whole-panel vector that
  // convert-vector-to-llvm can only lower by scalar-unrolling it.
  // tensor.pad needs explicit vector sizes; it fails to vectorize without.
  auto padTile =
      tileOneLevel(rewriter, hoistedPad.getOperation(), {kPackTileRows, 0});
  if (mlir::failed(padTile) || padTile->loops.size() != 1)
    return mlir::failure();
  rewriter.setInsertionPoint(padTile->op);
  // inputScalableVecDims must be given whenever inputVectorSizes is (the
  // vectorizer asserts on a length mismatch); nothing here is scalable.
  const llvm::SmallVector<bool> notScalable = {false, false};
  if (mlir::failed(mlir::linalg::vectorize(
          rewriter, padTile->op, /*inputVectorSizes=*/{kPackTileRows, s.nr},
          notScalable)))
    return mlir::failure();

  // Fold the insert_slice into the transfer_write *now*. Left alone, the
  // transfer_read/transfer_write round trip through a fresh tensor.empty
  // folds back to insert_slice(extract_slice(B)), which bufferizes to a
  // strided memref.copy — a call to the memrefCopy runtime helper, which is
  // an undefined symbol in the JIT.
  llvm::SmallVector<mlir::Operation *> packLoopOps;
  padTile->loops.front()->walk([&](mlir::Operation *op) {
    if (op != padTile->loops.front().getOperation())
      packLoopOps.push_back(op);
  });
  mlir::RewritePatternSet patterns(rewriter.getContext());
  mlir::tensor::populateFoldTensorSubsetIntoVectorTransferPatterns(patterns);
  return applyPatternSetTo(packLoopOps, std::move(patterns));
}

// A~ = (MC/MR) x KC x MR: hoists A's pad out of the register loops with a
// [1, 0] transpose, so each k step reads a contiguous MR-element row instead
// of MR scalars from MR different rows. Leaves an un-transpose in front of
// the tile for the kernel pass to fuse into its k-loop.
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
  mlir::linalg::TransposeOp packTranspose = transposeOps[0];

  // When no hoisted loop indexes A (MC == MR, so ir was folded and only jr is
  // hoisted out of), there is no packing loop and upstream's
  // replaceByPackingResult slices the untransposed hoisted pad instead of
  // the transpose's result (HoistPadding.cpp, LLVM 21). Point the slice
  // feeding the un-transpose at the transposed panel.
  auto unTransposeSrc = transposeOps[1]
                            .getInput()
                            .getDefiningOp<mlir::tensor::ExtractSliceOp>();
  if (unTransposeSrc && unTransposeSrc.getSource() == hoistedPad.getResult())
    rewriter.modifyOpInPlace(unTransposeSrc, [&] {
      unTransposeSrc.getSourceMutable().assign(packTranspose.getResult()[0]);
    });

  // A's pad is zero-width — the transpose is the packing copy — and
  // tensor.pad is not destination-style, so it cannot be fused with the
  // fusion utilities. Decomposing it lets the transpose read A directly.
  mlir::RewritePatternSet patterns(rewriter.getContext());
  mlir::linalg::populateDecomposePadPatterns(patterns);
  if (mlir::failed(
          applyPatternSetTo({hoistedPad.getOperation()}, std::move(patterns))))
    return mlir::failure();

  // Tiled to 8 rows for the same reason as B's copy; LinalgVectorizationPass
  // vectorizes the tiles later.
  auto trTile = tileOneLevel(rewriter, packTranspose.getOperation(),
                             {kPackTileRows, 0});
  return mlir::success(mlir::succeeded(trTile));
}

static bool isSliceOperand(mlir::Operation *tile, unsigned operand) {
  return tile->getOperand(operand)
      .getDefiningOp<mlir::tensor::ExtractSliceOp>();
}

// Packs one tiled tile's operands. Packing is skipped where it cannot pay:
// with no register loop left to hoist out of, each panel would be used by a
// single microtile; and an operand that is the whole original tensor (its
// full-size slice folded away) has nothing to gather — hoistPaddingOnTensors
// requires a slice to pack from.
static mlir::LogicalResult pack(mlir::IRRewriter &rewriter,
                                mlir::Operation *tile,
                                const mlir_edsl::MatmulStrategy &s) {
  llvm::SmallVector<mlir::scf::ForOp> hoistLoops = collectHoistLoops(tile);
  const int64_t hoistDepth = hoistLoops.size();
  for (mlir::scf::ForOp loop : hoistLoops)
    loop->removeAttr(mlir_edsl::kBlockedHoistAttrName);

  if (hoistDepth > 0 && (s.packA || s.packB)) {
    if (mlir::failed(mergeSliceChains(tile)))
      return mlir::failure();
    const bool withA = s.packA && isSliceOperand(tile, 0);
    const bool withB = s.packB && isSliceOperand(tile, 1);
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
    return "Pack the A and B operands of tiled blocked matmuls into "
           "contiguous panels hoisted out of the register loops";
  }

  void runOnOperation() override {
    mlir::func::FuncOp func = getOperation();
    mlir::IRRewriter rewriter(func->getContext());

    llvm::SmallVector<
        std::pair<mlir::linalg::MatmulOp, mlir_edsl::MatmulStrategy>>
        tiles;
    if (mlir::failed(mlir_edsl::collectBlockedTilesAtStage(
            func, BlockedStage::Tiled, tiles)))
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

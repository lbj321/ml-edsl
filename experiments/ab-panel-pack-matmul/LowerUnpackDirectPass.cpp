//===- LowerUnpackDirectPass.cpp -----------------------------------------===//
//
// Experimental (not yet wired into any CMake target - see the note in
// run.sh once this stage is added): an alternative to
// mlir::linalg::lowerUnPack for the one case that matters here - a
// linalg.unpack whose result feeds a single bufferization.
// materialize_in_destination and needs no padding remainder.
//
// The problem: lowerUnPack always allocates a fresh scratch tensor.empty for
// its internal linalg.transpose, then collapse_shapes and extract_slices the
// result before a final linalg.copy into the unpack's `dest`.
// -eliminate-empty-tensors can eliminate that trailing copy's own empty (it
// implements SubsetInsertionOpInterface's contract via linalg.copy), but it
// can never reach the *earlier* scratch empty behind the transpose, because
// neither tensor.collapse_shape nor tensor.expand_shape implement
// SubsetInsertionOpInterface/SubsetExtractionOpInterface - confirmed by
// reading BufferizationOps.td (only ops declaring that interface, like
// linalg.copy and bufferization.materialize_in_destination itself, are
// walked through during empty-tensor elimination) and by direct
// experimentation (see out/pack_lowered_eliminated.mlir in this directory:
// the outer empty gets swapped for %arg2, the inner one behind
// collapse_shape never does).
//
// The fix: skip the scratch empty entirely. Reshape the *real* destination
// buffer up to the transpose's pre-collapse (stripMined) shape via
// tensor.expand_shape, and have the transpose write directly into that view.
// Since tensor.expand_shape/collapse_shape are free metadata reshapes once
// bufferized, and there's no tensor.empty left in the chain for elimination
// to have to find, the transpose ends up writing straight into the real
// buffer with no eliminate-empty-tensors dependency at all.
//
// Naively reusing unpackOp.getDest() for this does NOT work - confirmed
// empirically (see out/ in this directory): that operand is almost always a
// dead, fill-initialized tensor.empty placeholder, and nothing upstream
// (not -eliminate-empty-tensors, not canonicalize) ever promotes it to the
// real destination on its own once it is only reachable through an
// expand_shape rather than a copy. So this pattern instead looks *forward*
// from the unpack's result to find its materialize_in_destination consumer
// and builds `bufferization.to_tensor %thatMemref restrict writable`
// directly, discarding the original placeholder dest. Validated against the
// full pack -> tile(scf.forall) -> lower-unpack -> bufferize chain in this
// experiment: produces zero memref.copy into the destination, and the
// leftover self-copy inside the forall body (from tile_using_forall's own
// bufferization of tensor.parallel_insert_slice) cleans up under a normal
// trailing canonicalize pass - both confirmed empirically, not assumed.
//
// Only handles the case above; anything else (multiple uses, no
// materialize_in_destination consumer, or a padding remainder where dest is
// smaller than the strip-mined shape and can't be expand_shape'd up to it)
// falls back to the stock mlir::linalg::lowerUnPack unchanged.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;

namespace {

/// Returns unpackResult's sole materialize_in_destination user, or null if
/// unpackResult has more than one use or its one use isn't that op.
bufferization::MaterializeInDestinationOp
getSoleMaterializeInDestinationUser(Value unpackResult) {
  if (!unpackResult.hasOneUse())
    return nullptr;
  return dyn_cast<bufferization::MaterializeInDestinationOp>(
      (*unpackResult.getUses().begin()).getOwner());
}

/// Lowers a linalg.unpack directly into its destination buffer, bypassing
/// mlir::linalg::lowerUnPack's scratch tensor.empty. See file comment.
struct LowerUnpackDirectPattern : public OpRewritePattern<linalg::UnPackOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::UnPackOp unpackOp,
                                 PatternRewriter &rewriter) const override {
    auto materializeOp =
        getSoleMaterializeInDestinationUser(unpackOp.getResult());
    if (!materializeOp)
      return rewriter.notifyMatchFailure(
          unpackOp, "expected unpack's sole use to be a "
                    "materialize_in_destination into a memref");
    if (!isa<MemRefType>(materializeOp.getDest().getType()))
      return rewriter.notifyMatchFailure(
          materializeOp, "expected materialize_in_destination's dest to "
                         "already be a memref");

    Location loc = unpackOp.getLoc();

    PackingMetadata packingMetadata;
    SmallVector<int64_t> packedToStripMinedShapePerm =
        linalg::getUnPackInverseSrcPerm(unpackOp, packingMetadata);

    RankedTensorType packedTensorType = unpackOp.getSourceType();
    SmallVector<int64_t> stripMinedShape(packedTensorType.getShape());
    applyPermutationToVector(stripMinedShape, packedToStripMinedShapePerm);
    auto stripMinedTensorType =
        RankedTensorType::Builder(packedTensorType).setShape(stripMinedShape);
    auto collapsedType = tensor::CollapseShapeOp::inferCollapsedType(
        stripMinedTensorType, packingMetadata.reassociations);

    auto destTensorType = cast<RankedTensorType>(unpackOp.getDestType());
    if (collapsedType.getShape() != destTensorType.getShape())
      return rewriter.notifyMatchFailure(
          unpackOp, "destination has a padding remainder smaller than the "
                    "strip-mined shape; falling back to stock lowerUnPack");

    rewriter.setInsertionPoint(unpackOp);
    Value realDest = rewriter.create<bufferization::ToTensorOp>(
        loc, materializeOp.getDest(), /*restrict=*/true, /*writeable=*/true);
    Value expandedDest = rewriter.create<tensor::ExpandShapeOp>(
        loc, stripMinedTensorType, realDest, packingMetadata.reassociations);
    Value transposed =
        rewriter
            .create<linalg::TransposeOp>(loc, unpackOp.getSource(),
                                          expandedDest,
                                          packedToStripMinedShapePerm)
            ->getResult(0);
    Value collapsed = rewriter.create<tensor::CollapseShapeOp>(
        loc, destTensorType, transposed, packingMetadata.reassociations);

    rewriter.replaceOp(unpackOp, collapsed);
    // materializeOp now copies realDest's own data back into itself; a
    // trailing canonicalize pass folds that away (confirmed empirically -
    // see out/ in this directory).
    return success();
  }
};

struct LowerUnpackDirectPass
    : public PassWrapper<LowerUnpackDirectPass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerUnpackDirectPass)

  StringRef getArgument() const override {
    return "linalg-lower-unpack-direct";
  }
  StringRef getDescription() const override {
    return "Experimental: lower linalg.unpack directly into its destination "
           "buffer instead of through a scratch tensor.empty, falling back "
           "to the stock decomposition where that isn't safe";
  }

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    RewritePatternSet patterns(func.getContext());
    patterns.add<LowerUnpackDirectPattern>(func.getContext());
    if (failed(applyPatternsGreedily(func, std::move(patterns))))
      signalPassFailure();

    // Anything the pattern above declined (padding remainder, no direct
    // materialize_in_destination consumer, etc.) still needs to be lowered
    // somehow - fall back to the stock decomposition for those.
    IRRewriter rewriter(func.getContext());
    SmallVector<linalg::UnPackOp> remaining;
    func.walk([&](linalg::UnPackOp op) { remaining.push_back(op); });
    for (linalg::UnPackOp op : remaining) {
      if (failed(linalg::lowerUnPack(rewriter, op)))
        op->emitWarning(
            "linalg-lower-unpack-direct: fallback lowering failed");
    }
  }
};

} // namespace

std::unique_ptr<Pass> createLowerUnpackDirectPass() {
  return std::make_unique<LowerUnpackDirectPass>();
}

#include "TilingUtils.h"

#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/Transforms/Transforms.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/Interfaces/TilingInterface.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir_edsl {

mlir::LogicalResult applyPatternSetTo(llvm::ArrayRef<mlir::Operation *> ops,
                                      mlir::RewritePatternSet &&patterns) {
  mlir::GreedyRewriteConfig config;
  config.setStrictness(mlir::GreedyRewriteStrictness::ExistingAndNewOps);
  return mlir::applyOpPatternsGreedily(ops, std::move(patterns), config);
}

mlir::LogicalResult vectorizeCopyTile(mlir::IRRewriter &rewriter,
                                      mlir::Operation *tile,
                                      mlir::Operation *loop,
                                      llvm::ArrayRef<int64_t> vectorSizes) {
  rewriter.setInsertionPoint(tile);
  // inputScalableVecDims must be given whenever inputVectorSizes is (the
  // vectorizer asserts on a length mismatch); nothing here is scalable.
  const llvm::SmallVector<bool> notScalable(vectorSizes.size(), false);
  if (mlir::failed(
          mlir::linalg::vectorize(rewriter, tile, vectorSizes, notScalable)))
    return mlir::failure();

  llvm::SmallVector<mlir::Operation *> loopOps;
  loop->walk([&](mlir::Operation *op) {
    if (op != loop)
      loopOps.push_back(op);
  });
  mlir::RewritePatternSet patterns(rewriter.getContext());
  mlir::tensor::populateFoldTensorSubsetIntoVectorTransferPatterns(patterns);
  return applyPatternSetTo(loopOps, std::move(patterns));
}

mlir::FailureOr<TiledLevel>
tileOneLevel(mlir::IRRewriter &rewriter, mlir::Operation *op,
             llvm::ArrayRef<int64_t> sizes,
             mlir::scf::SCFTilingOptions::LoopType loopType) {
  auto tilingOp = llvm::dyn_cast<mlir::TilingInterface>(op);
  if (!tilingOp)
    return mlir::failure();

  // Named variable required — setTileSizes captures a non-owning ArrayRef.
  llvm::SmallVector<mlir::OpFoldResult> tileSizes =
      mlir::getAsIndexOpFoldResult(op->getContext(), sizes);
  mlir::scf::SCFTilingOptions opts;
  opts.setTileSizes(tileSizes);
  opts.setLoopType(loopType);

  rewriter.setInsertionPoint(op);
  auto result = mlir::scf::tileUsingSCF(rewriter, tilingOp, opts);
  if (mlir::failed(result))
    return mlir::failure();
  if (result->tiledOps.size() != 1)
    return mlir::failure();

  rewriter.replaceOp(op, result->mergeResult.replacements);
  return TiledLevel{result->tiledOps.front(), std::move(result->loops)};
}

} // namespace mlir_edsl

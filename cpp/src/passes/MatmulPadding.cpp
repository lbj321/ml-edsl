#include "MatmulPadding.h"

#include "TilingUtils.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"

#include <array>

namespace mlir_edsl {

namespace {

// Largest of `candidates` (descending) dividing `extent`, so every tile is
// static and LinalgVectorizationPass can vectorize it; 1 always divides.
int64_t largestDividingTile(int64_t extent,
                            llvm::ArrayRef<int64_t> candidates) {
  for (int64_t c : candidates)
    if (extent % c == 0)
      return c;
  return 1;
}

// Tiles a 2-D copy or fill into a parallel scf.forall of row strips, each a
// serial loop over tiles of up to 8 x 16, and vectorizes the tiles. Left
// whole, the op would reach LinalgVectorizationPass as one M x N vector.
mlir::LogicalResult tileCopyLike(mlir::IRRewriter &rewriter,
                                 mlir::Operation *op) {
  // Tiling moves the insertion point into the loops it builds.
  mlir::OpBuilder::InsertionGuard guard(rewriter);
  auto shape =
      llvm::cast<mlir::RankedTensorType>(op->getResult(0).getType()).getShape();
  const int64_t rows = largestDividingTile(shape[0], {8, 4, 2, 1});
  const int64_t cols = largestDividingTile(shape[1], {16, 8, 4, 2, 1});
  auto strips = tileOneLevel(rewriter, op, {rows, 0},
                             mlir::scf::SCFTilingOptions::LoopType::ForallOp);
  if (mlir::failed(strips))
    return mlir::failure();
  auto tiles = tileOneLevel(rewriter, strips->op, {0, cols});
  if (mlir::failed(tiles) || tiles->loops.size() != 1)
    return mlir::failure();
  return vectorizeCopyTile(rewriter, tiles->op,
                           tiles->loops.front().getOperation());
}

llvm::SmallVector<mlir::OpFoldResult> indices(mlir::MLIRContext *ctx,
                                              llvm::ArrayRef<int64_t> v) {
  return mlir::getAsIndexOpFoldResult(ctx, v);
}

// `dest` with `op`'s result written into its [r0, c0] corner. The op must
// already write into an extract_slice of `dest` at the same place.
mlir::Value insertAt(mlir::IRRewriter &rewriter, mlir::Location loc,
                     mlir::Operation *op, mlir::Value dest, int64_t r0,
                     int64_t c0) {
  mlir::MLIRContext *ctx = rewriter.getContext();
  auto type = llvm::cast<mlir::RankedTensorType>(op->getResult(0).getType());
  return rewriter.create<mlir::tensor::InsertSliceOp>(
      loc, op->getResult(0), dest, indices(ctx, {r0, c0}),
      indices(ctx, type.getShape()), indices(ctx, {1, 1}));
}

mlir::Value sliceOf(mlir::IRRewriter &rewriter, mlir::Location loc,
                    mlir::Value src, int64_t r0, int64_t c0, int64_t h,
                    int64_t w) {
  mlir::MLIRContext *ctx = rewriter.getContext();
  return rewriter.create<mlir::tensor::ExtractSliceOp>(
      loc, src, indices(ctx, {r0, c0}), indices(ctx, {h, w}),
      indices(ctx, {1, 1}));
}

// `src` (r x c) zero-padded to rp x cp. Only the pad strips are zeroed; the
// rest is overwritten by the copy of `src`.
mlir::FailureOr<mlir::Value> padOperand(mlir::IRRewriter &rewriter,
                                        mlir::Location loc, mlir::Value src,
                                        int64_t rp, int64_t cp) {
  auto type = llvm::cast<mlir::RankedTensorType>(src.getType());
  const int64_t r = type.getDimSize(0);
  const int64_t c = type.getDimSize(1);
  if (r == rp && c == cp)
    return src;

  mlir::Type elementType = type.getElementType();
  mlir::Value padded = rewriter.create<mlir::tensor::EmptyOp>(
      loc, llvm::ArrayRef<int64_t>{rp, cp}, elementType);
  mlir::Value zero = rewriter.create<mlir::arith::ConstantOp>(
      loc, rewriter.getZeroAttr(elementType));

  llvm::SmallVector<mlir::Operation *> toTile;
  // Right strip beside the copy, then the bottom strip across the full width.
  const std::array<std::array<int64_t, 4>, 2> strips = {{
      {0, c, r, cp - c},
      {r, 0, rp - r, cp},
  }};
  for (auto [r0, c0, h, w] : strips) {
    if (h == 0 || w == 0)
      continue;
    auto fill = rewriter.create<mlir::linalg::FillOp>(
        loc, mlir::ValueRange{zero},
        mlir::ValueRange{sliceOf(rewriter, loc, padded, r0, c0, h, w)});
    padded = insertAt(rewriter, loc, fill, padded, r0, c0);
    toTile.push_back(fill);
  }

  auto copy = rewriter.create<mlir::linalg::CopyOp>(
      loc, src, sliceOf(rewriter, loc, padded, 0, 0, r, c));
  padded = insertAt(rewriter, loc, copy, padded, 0, 0);
  toTile.push_back(copy);

  for (mlir::Operation *op : toTile)
    if (mlir::failed(tileCopyLike(rewriter, op)))
      return mlir::failure();
  return padded;
}

} // namespace

mlir::FailureOr<mlir::linalg::MatmulOp>
padToBlocks(mlir::IRRewriter &rewriter, mlir::linalg::MatmulOp op,
            const MatmulStrategy &s, bool &committed) {
  committed = false;
  mlir::Value a = op.getInputs()[0];
  mlir::Value b = op.getInputs()[1];
  mlir::Value c = op.getOutputs()[0];
  auto cType = llvm::cast<mlir::RankedTensorType>(c.getType());
  const int64_t m = cType.getDimSize(0);
  const int64_t n = cType.getDimSize(1);
  const int64_t k =
      llvm::cast<mlir::RankedTensorType>(a.getType()).getDimSize(1);
  const int64_t mp = paddedExtent(m, s.mc);
  const int64_t np = paddedExtent(n, s.nc);
  const int64_t kp = paddedExtent(k, s.kc);
  if (mp == m && np == n && kp == k)
    return op;
  // A transposed or broadcast matmul would need its operands padded along
  // other dimensions than the ones read off here.
  if (op.hasUserDefinedMaps())
    return mlir::failure();

  committed = true;
  mlir::Location loc = op.getLoc();
  rewriter.setInsertionPoint(op);

  auto paddedA = padOperand(rewriter, loc, a, mp, kp);
  auto paddedB = padOperand(rewriter, loc, b, kp, np);
  if (mlir::failed(paddedA) || mlir::failed(paddedB))
    return mlir::failure();

  // The EDSL's C is a zero fill; filling the padded C directly avoids
  // copying one fill into another.
  mlir::Value paddedC;
  if (auto fill = c.getDefiningOp<mlir::linalg::FillOp>()) {
    mlir::Value empty = rewriter.create<mlir::tensor::EmptyOp>(
        loc, llvm::ArrayRef<int64_t>{mp, np}, cType.getElementType());
    paddedC = rewriter
                  .create<mlir::linalg::FillOp>(loc, fill.getInputs(),
                                                mlir::ValueRange{empty})
                  .getResult(0);
  } else {
    auto paddedInit = padOperand(rewriter, loc, c, mp, np);
    if (mlir::failed(paddedInit))
      return mlir::failure();
    paddedC = *paddedInit;
  }

  auto padded = rewriter.create<mlir::linalg::MatmulOp>(
      loc, mlir::TypeRange{paddedC.getType()},
      mlir::ValueRange{*paddedA, *paddedB}, mlir::ValueRange{paddedC});

  mlir::Value result = padded.getResult(0);
  if (mp != m || np != n) {
    mlir::Value empty = rewriter.create<mlir::tensor::EmptyOp>(
        loc, llvm::ArrayRef<int64_t>{m, n}, cType.getElementType());
    auto copyOut = rewriter.create<mlir::linalg::CopyOp>(
        loc, sliceOf(rewriter, loc, result, 0, 0, m, n), empty);
    result = copyOut.getResult(0);
    rewriter.replaceOp(op, result);
    if (mlir::failed(tileCopyLike(rewriter, copyOut)))
      return mlir::failure();
  } else {
    rewriter.replaceOp(op, result);
  }
  return padded;
}

} // namespace mlir_edsl

//===- LinalgMatmulBlockedKernel.cpp - k-loop over the register tile ------===//
//
// linalg-matmul-blocked-kernel: the last of the three blocked matmul passes
// (see BlockedStage in MatmulStrategy.h). Tiles each Tiled MR x NR x KC tile
// over k into the MR x NR x 1 register tile the microkernel passes build on,
// and leaves it at stage Kernel.
//
//===----------------------------------------------------------------------===//

#include "TilingUtils.h"

#include "mlir_edsl/MLIRLoweringPasses.h"
#include "mlir_edsl/MatmulStrategy.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"

namespace {

using mlir_edsl::BlockedStage;

// Reduces the MR x NR x KC tile to the MR x NR x 1 register tile.
//
// When A was packed there is an un-transpose in front of the tile (hoisting
// with a transpose leaves one behind, see packA in the tile-and-pack pass)
// which must move inside this loop, or every microtile pays a full MR x KC
// copy. Fusing the producer into the generated k-loop makes each k step
// transpose a 1 x MR row of A~ instead — i.e. exactly the contiguous read the
// packing was for. The un-transpose is looked for rather than inferred from
// the strategy, since packing can be skipped for a tile.
static mlir::FailureOr<mlir::Operation *>
tileK(mlir::IRRewriter &rewriter, mlir::Operation *tile) {
  if (!tile->getOperand(0).getDefiningOp<mlir::linalg::TransposeOp>()) {
    auto tiled = mlir_edsl::tileOneLevel(rewriter, tile, {0, 0, 1});
    if (mlir::failed(tiled))
      return mlir::failure();
    return tiled->op;
  }

  auto tilingOp = llvm::dyn_cast<mlir::TilingInterface>(tile);
  if (!tilingOp)
    return mlir::failure();

  llvm::SmallVector<mlir::OpFoldResult> tileSizes =
      mlir::getAsIndexOpFoldResult(rewriter.getContext(), {0, 0, 1});
  mlir::scf::SCFTileAndFuseOptions opts;
  opts.tilingOptions.setTileSizes(tileSizes);
  rewriter.setInsertionPoint(tile);
  auto result =
      mlir::scf::tileConsumerAndFuseProducersUsingSCF(rewriter, tilingOp, opts);
  if (mlir::failed(result))
    return mlir::failure();

  llvm::SmallVector<mlir::Value> replacements;
  for (mlir::Value res : tile->getResults())
    replacements.push_back(result->replacements.lookup(res));
  rewriter.replaceOp(tile, replacements);

  // tiledAndFusedOps also holds the fused un-transpose.
  auto kernel = llvm::find_if(result->tiledAndFusedOps, [](mlir::Operation *op) {
    return llvm::isa<mlir::linalg::MatmulOp>(op);
  });
  if (kernel == result->tiledAndFusedOps.end())
    return mlir::failure();
  return *kernel;
}

struct LinalgMatmulBlockedKernelPass
    : public mlir::PassWrapper<LinalgMatmulBlockedKernelPass,
                               mlir::OperationPass<mlir::func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LinalgMatmulBlockedKernelPass)

  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::scf::SCFDialect, mlir::tensor::TensorDialect,
                    mlir::linalg::LinalgDialect>();
  }

  llvm::StringRef getArgument() const override {
    return "linalg-matmul-blocked-kernel";
  }
  llvm::StringRef getDescription() const override {
    return "Tile packed blocked matmuls over k into the MR x NR x 1 register "
           "tile the microkernel passes build on";
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
      auto kernel = tileK(rewriter, tile);
      if (mlir::failed(kernel)) {
        func->emitError("linalg-matmul-blocked-kernel failed");
        return signalPassFailure();
      }
      mlir_edsl::setBlockedStage(*kernel, BlockedStage::Kernel);
    }
  }
};

} // namespace

namespace mlir_edsl {

std::unique_ptr<mlir::Pass> createLinalgMatmulBlockedKernelPass() {
  return std::make_unique<LinalgMatmulBlockedKernelPass>();
}

} // namespace mlir_edsl

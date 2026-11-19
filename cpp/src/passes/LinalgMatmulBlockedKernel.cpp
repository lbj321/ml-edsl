//===- LinalgMatmulBlockedKernel.cpp - k-loop over the register tile ------===//
//
// linalg-matmul-blocked-kernel: the third of the six blocked matmul passes
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
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"

namespace {

using mlir_edsl::BlockedStage;

// Reduces the MR x NR x KC tile to the MR x NR x 1 register tile, marking
// the k-loop for pack-b and pack-a to hoist out of.
static mlir::FailureOr<mlir::Operation *>
tileK(mlir::IRRewriter &rewriter, mlir::Operation *tile) {
  auto tiled = mlir_edsl::tileOneLevel(rewriter, tile, {0, 0, 1});
  if (mlir::failed(tiled) || tiled->loops.size() != 1)
    return mlir::failure();
  tiled->loops.front()->setAttr(mlir_edsl::kBlockedHoistAttrName,
                                rewriter.getUnitAttr());
  return tiled->op;
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
    return "Tile tiled blocked matmuls over k into the MR x NR x 1 register "
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

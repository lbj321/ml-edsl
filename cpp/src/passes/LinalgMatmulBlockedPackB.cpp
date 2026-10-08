//===- LinalgMatmulBlockedPackB.cpp - pack B into contiguous panels -------===//
//
// linalg-matmul-blocked-pack-b: the fifth of the six blocked matmul passes
// (see BlockedStage in MatmulStrategy.h). Hoists the nofold pad pack-prepare
// put on each Padded tile's B out of the loops marked with
// kBlockedHoistAttrName, which packs B into contiguous panels. The tile stays
// at stage Padded, and the markers stay for pack-a.
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
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"

namespace {

using mlir_edsl::BlockedStage;

// B~ = (NC/NR) x KC x 1 x NR: hoists B's pad out of the marked loops with no
// transpose. The pad is a plain row copy, so the packed panel keeps B's (k, n)
// order and the microkernel reads contiguous NR-wide rows instead of striding
// by N. A tile without a nofold pad on B is left alone.
static mlir::LogicalResult packB(mlir::IRRewriter &rewriter,
                                 mlir::Operation *tile,
                                 const mlir_edsl::MatmulStrategy &s) {
  mlir::tensor::PadOp padB = mlir_edsl::nofoldPadOperand(tile, 1);
  if (!padB)
    return mlir::success();

  const int64_t hoistDepth = mlir_edsl::collectHoistLoops(tile).size();
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
  // This must run before any canonicalize; see vectorizeCopyTile.
  return mlir_edsl::vectorizeCopyTile(rewriter, hoistedPad.getOperation(),
                                      hoistedPad->getParentOp(),
                                      /*vectorSizes=*/{1, s.nr});
}

struct LinalgMatmulBlockedPackBPass
    : public mlir::PassWrapper<LinalgMatmulBlockedPackBPass,
                               mlir::OperationPass<mlir::func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LinalgMatmulBlockedPackBPass)

  // Hoisting builds scf.for packing loops and tensor.empty; vectorizing the
  // row copy builds vector transfers.
  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::scf::SCFDialect, mlir::tensor::TensorDialect,
                    mlir::linalg::LinalgDialect, mlir::vector::VectorDialect,
                    mlir::arith::ArithDialect>();
  }

  llvm::StringRef getArgument() const override {
    return "linalg-matmul-blocked-pack-b";
  }
  llvm::StringRef getDescription() const override {
    return "Pack the B operand of padded blocked matmuls into contiguous "
           "panels hoisted out of the k and register loops";
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
      if (mlir::failed(packB(rewriter, tile, strategy))) {
        func->emitError("linalg-matmul-blocked-pack-b failed");
        return signalPassFailure();
      }
    }
  }
};

} // namespace

namespace mlir_edsl {

std::unique_ptr<mlir::Pass> createLinalgMatmulBlockedPackBPass() {
  return std::make_unique<LinalgMatmulBlockedPackBPass>();
}

} // namespace mlir_edsl

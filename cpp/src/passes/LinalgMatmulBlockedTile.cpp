//===- LinalgMatmulBlockedTile.cpp - cache and register tiles -------------===//
//
// linalg-matmul-blocked-tile: the second of the four blocked matmul passes
// (see BlockedStage in MatmulStrategy.h). Tiles each Distributed tile into
// pc → jr → ir loops over an MR x NR x KC tile, marks jr and ir for the pack
// pass to hoist out of, and leaves the tile at stage Tiled.
//
//===----------------------------------------------------------------------===//

#include "TilingUtils.h"

#include "mlir_edsl/MLIRLoweringPasses.h"
#include "mlir_edsl/MatmulStrategy.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"

#include <array>

namespace {

using mlir_edsl::BlockedStage;

// Tiles a distributed MC x NC x K tile into
//
//   pc (K/KC) → jr (NC/NR) → ir (MC/MR)
//
// leaving an MR x NR x KC tile. The rest stay serial: K must not be split
// across threads (it is the reduction), and the register tiles are
// per-thread work.
static mlir::LogicalResult tileCacheAndRegisterLevels(
    mlir::IRRewriter &rewriter, mlir::Operation *tile,
    const mlir_edsl::MatmulStrategy &s) {
  struct Level {
    std::array<int64_t, 3> sizes;
    bool hoistOut; // a register loop the packed panels are hoisted out of
  };
  const std::array<Level, 3> levels = {{
      {{0, 0, s.kc}, false}, // pc: K cache block
      {{0, s.nr, 0}, true},  // jr: N register tile
      {{s.mr, 0, 0}, true},  // ir: M register tile
  }};

  for (const Level &level : levels) {
    auto tiled = mlir_edsl::tileOneLevel(rewriter, tile, level.sizes);
    if (mlir::failed(tiled) || tiled->loops.size() != 1)
      return mlir::failure();
    tile = tiled->op;
    if (level.hoistOut)
      tiled->loops.front()->setAttr(mlir_edsl::kBlockedHoistAttrName,
                                    rewriter.getUnitAttr());
  }

  mlir_edsl::setBlockedStage(tile, BlockedStage::Tiled);
  return mlir::success();
}

struct LinalgMatmulBlockedTilePass
    : public mlir::PassWrapper<LinalgMatmulBlockedTilePass,
                               mlir::OperationPass<mlir::func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LinalgMatmulBlockedTilePass)

  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::scf::SCFDialect, mlir::tensor::TensorDialect,
                    mlir::linalg::LinalgDialect, mlir::arith::ArithDialect>();
  }

  llvm::StringRef getArgument() const override {
    return "linalg-matmul-blocked-tile";
  }
  llvm::StringRef getDescription() const override {
    return "Tile distributed blocked matmuls into pc/jr/ir loops over an "
           "MR x NR x KC tile";
  }

  void runOnOperation() override {
    mlir::func::FuncOp func = getOperation();
    mlir::IRRewriter rewriter(func->getContext());

    llvm::SmallVector<
        std::pair<mlir::linalg::MatmulOp, mlir_edsl::MatmulStrategy>>
        tiles;
    if (mlir::failed(mlir_edsl::collectBlockedTilesAtStage(
            func, BlockedStage::Distributed, tiles)))
      return signalPassFailure();

    // A tile left half-done here still carries mlir_edsl.blocked, which keeps
    // the older passes off it, so a failure cannot fall back to them.
    for (auto &[tile, strategy] : tiles) {
      if (mlir::failed(tileCacheAndRegisterLevels(rewriter, tile, strategy))) {
        func->emitError("linalg-matmul-blocked-tile failed");
        return signalPassFailure();
      }
    }
  }
};

} // namespace

namespace mlir_edsl {

std::unique_ptr<mlir::Pass> createLinalgMatmulBlockedTilePass() {
  return std::make_unique<LinalgMatmulBlockedTilePass>();
}

} // namespace mlir_edsl

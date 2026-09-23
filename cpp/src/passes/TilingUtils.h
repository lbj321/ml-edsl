#pragma once

#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "llvm/ADT/SmallVector.h"

#include <cstdint>

namespace mlir_edsl {

/// The tile op one tiling level produced and the loops generated around it,
/// outermost first.
struct TiledLevel {
  mlir::Operation *op;
  llvm::SmallVector<mlir::LoopLikeOpInterface> loops;
};

/// Tiles one level of a loop nest, replacing `op` with the tiled result.
/// Exactly one entry of `sizes` is expected to be non-zero per call when
/// building a nest level by level: issuing one call per level is what fixes
/// the resulting loop order, which a single multi-dimensional tile_using_for
/// would not (see PLAN.md Stage 2 — [6,16,0] gives ir-outer/jr-inner, the
/// opposite of what the macro-kernel wants).
mlir::FailureOr<TiledLevel>
tileOneLevel(mlir::IRRewriter &rewriter, mlir::Operation *op,
             llvm::ArrayRef<int64_t> sizes,
             mlir::scf::SCFTilingOptions::LoopType loopType =
                 mlir::scf::SCFTilingOptions::LoopType::ForOp);

} // namespace mlir_edsl

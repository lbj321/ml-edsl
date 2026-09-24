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

/// Runs one pattern set greedily over `ops` only, plus the ops the patterns
/// create. Scoping is what lets the caller keep holding handles: ops outside
/// the set are never folded, rewritten or erased as dead. Kept deliberately
/// narrow — no canonicalizer — because canonicalization at the wrong moment
/// undoes a vectorized copy (see vectorizeCopyTile).
mlir::LogicalResult applyPatternSetTo(llvm::ArrayRef<mlir::Operation *> ops,
                                      mlir::RewritePatternSet &&patterns);

/// Vectorizes `tile`, a static-shape copy, pad or fill tile inside `loop`, and
/// folds the insert_slice storing it into the vector.transfer_write, scoped to
/// the ops in `loop`. Left for later, canonicalize folds the transfer_read /
/// transfer_write round trip back to insert_slice(extract_slice), which
/// bufferizes to a strided memref.copy: a call to the memrefCopy runtime
/// helper, a generic element-by-element loop.
mlir::LogicalResult vectorizeCopyTile(mlir::IRRewriter &rewriter,
                                      mlir::Operation *tile,
                                      mlir::Operation *loop,
                                      llvm::ArrayRef<int64_t> vectorSizes = {});

} // namespace mlir_edsl

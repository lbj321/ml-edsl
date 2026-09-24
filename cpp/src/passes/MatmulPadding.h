#pragma once

#include "mlir_edsl/MatmulStrategy.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/PatternMatch.h"

namespace mlir_edsl {

/// Pads `op` so every extent is a multiple of its cache block in `s`: the
/// operands are copied into zero-padded tensors, the matmul runs on those, and
/// the result is copied back out of the padded C. Only the extents that need
/// it are padded; when none do, `op` is returned untouched.
///
/// The copies and the zero fills are tiled here, into an scf.forall of row
/// strips over static tiles, because no later pass tiles a linalg.copy or a
/// fill left whole. `committed` is set once the IR has been changed, so the
/// caller can tell a failure that left `op` intact from one that did not.
mlir::FailureOr<mlir::linalg::MatmulOp>
padToBlocks(mlir::IRRewriter &rewriter, mlir::linalg::MatmulOp op,
            const MatmulStrategy &s, bool &committed);

} // namespace mlir_edsl

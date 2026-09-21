#pragma once

#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/StringRef.h"
#include <memory>

namespace mlir_edsl {

std::unique_ptr<mlir::Pass> createLinalgOuterTileAndFusePass(
    int64_t tileM = 64, int64_t tileN = 64);
std::unique_ptr<mlir::Pass> createLinalgMatmulToContractPass();
std::unique_ptr<mlir::Pass> createLinalgVectorizationPass();
std::unique_ptr<mlir::Pass> createVectorCleanupPass();

/// Lower permuting vector transfers and vector.transpose to shuffles.
std::unique_ptr<mlir::Pass> createVectorTransposeLoweringPass();
std::unique_ptr<mlir::Pass> createAllocaScopeCleanupPass();
std::unique_ptr<mlir::Pass> createVectorContractToOuterProductPass();
std::unique_ptr<mlir::Pass> createLinalgGenericTilingPass();
std::unique_ptr<mlir::Pass> createLinalgMatmulTilingPass();
std::unique_ptr<mlir::Pass> createLinalgMatmulParallelTilingPass();
std::unique_ptr<mlir::Pass> createLinalgMatmulKTilingPass();

/// BLIS-style cache and register blocking (jc/pc/ic/jr/ir over an MR x NR
/// register tile). Configured through pass options — see MatmulStrategy.h.
/// Runs first in buildCPUPipeline with its default options; the options
/// themselves are reachable only from mlir-edsl-opt.
std::unique_ptr<mlir::Pass> createLinalgMatmulBlockedPass();

#ifdef MLIR_EDSL_CUDA_ENABLED
std::unique_ptr<mlir::Pass> createLinalgGPUMatmulTilingPass();
#endif

} // namespace mlir_edsl

#pragma once

#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/StringRef.h"
#include <memory>

namespace mlir_edsl {

// Parse `strategy` eagerly using `ctx` (which must already have transform
// dialect extensions registered) and return a pass that applies it.
// If `guardLibraryCall` is non-empty, the pass is a no-op on modules that
// don't contain a linalg.generic with that library_call attribute.
std::unique_ptr<mlir::Pass> createTransformStrategyPass(
    mlir::MLIRContext *ctx, llvm::StringRef strategy,
    llvm::StringRef guardLibraryCall = "");
std::unique_ptr<mlir::Pass> createLinalgOuterTileAndFusePass();
std::unique_ptr<mlir::Pass> createLinalgMatmulToContractPass();
std::unique_ptr<mlir::Pass> createLinalgVectorizationPass();
std::unique_ptr<mlir::Pass> createVectorCleanupPass();
std::unique_ptr<mlir::Pass> createVectorContractToOuterProductPass();
std::unique_ptr<mlir::Pass> createLinalgGenericTilingPass();
std::unique_ptr<mlir::Pass> createLinalgMatmulTilingPass();
std::unique_ptr<mlir::Pass> createLinalgMatmulParallelTilingPass();

#ifdef MLIR_EDSL_CUDA_ENABLED
std::unique_ptr<mlir::Pass> createLinalgGPUMatmulTilingPass();
#endif

} // namespace mlir_edsl

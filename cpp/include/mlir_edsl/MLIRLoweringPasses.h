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
std::unique_ptr<mlir::Pass> createAllocaScopeCleanupPass();
std::unique_ptr<mlir::Pass> createVectorContractToOuterProductPass();
std::unique_ptr<mlir::Pass> createLinalgGenericTilingPass();
std::unique_ptr<mlir::Pass> createLinalgMatmulTilingPass();
std::unique_ptr<mlir::Pass> createLinalgMatmulParallelTilingPass();
std::unique_ptr<mlir::Pass> createLinalgMatmulKTilingPass();

#ifdef MLIR_EDSL_CUDA_ENABLED
std::unique_ptr<mlir::Pass> createLinalgGPUMatmulTilingPass();
#endif

} // namespace mlir_edsl

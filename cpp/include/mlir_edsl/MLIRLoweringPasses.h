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

// Experimental: not yet wired into addCPUPasses.
std::unique_ptr<mlir::Pass> createLinalgMatmulPackPass(
    int64_t packM = 32, int64_t packN = 32, int64_t packK = 32);
std::unique_ptr<mlir::Pass> createLinalgMatmulPackedForallTilingPass();
std::unique_ptr<mlir::Pass> createLinalgMatmulPackedKTilingPass();
std::unique_ptr<mlir::Pass> createLinalgMatmulPackedInnerTilingPass();
std::unique_ptr<mlir::Pass> createLinalgMatmulPackedVectorizePass();
std::unique_ptr<mlir::Pass> createLinalgMatmulPackedReductionToContractPass();
std::unique_ptr<mlir::Pass> createLinalgMatmulPackedLowerPackPass();
std::unique_ptr<mlir::Pass> createLinalgMatmulPackedLowerUnpackPass();

#ifdef MLIR_EDSL_CUDA_ENABLED
std::unique_ptr<mlir::Pass> createLinalgGPUMatmulTilingPass();
#endif

} // namespace mlir_edsl

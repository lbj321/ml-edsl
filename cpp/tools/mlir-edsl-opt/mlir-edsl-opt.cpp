// mlir-edsl-opt: standalone driver for the project's own MLIR passes,
// analogous to upstream `mlir-opt`. Lets a pass (or the full CPU lowering
// pipeline) be run directly on textual .mlir input/output, without going
// through the Python -> protobuf -> JIT path — useful for isolating a pass
// that misbehaves only as part of the real pipeline (e.g. under ASan).
//
// Dialect registration and the CPU pass sequence are shared with the JIT
// lowering path via mlir_edsl::registerCPUDialects/buildCPUPipeline
// (cpp/include/mlir_edsl/MLIRLowering.h), so this tool can't silently drift
// out of sync with what MLIRLowering::lowerToLLVMModule actually runs.

#include "mlir_edsl/MLIRLowering.h"
#include "mlir_edsl/MLIRLoweringPasses.h"

#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/SCF/Transforms/Passes.h"
#include "mlir/Dialect/Vector/Transforms/Passes.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "mlir/Transforms/Passes.h"

using namespace mlir_edsl;

namespace {

void registerCPUPasses() {
  mlir::registerPass(
      [] { return createLinalgOuterTileAndFusePass(); });
  mlir::registerPass([] { return createLinalgMatmulToContractPass(); });
  mlir::registerPass([] { return createLinalgVectorizationPass(); });
  mlir::registerPass([] { return createVectorCleanupPass(); });
  mlir::registerPass([] { return createAllocaScopeCleanupPass(); });
  mlir::registerPass(
      [] { return createVectorContractToOuterProductPass(); });
  // LinalgMatmulTilingPass backs two distinct pass-argument names depending
  // on its (tileM, tileN, tileK, loopType) configuration — see getArgument()
  // in MLIRLoweringPasses.cpp — so both factories must be registered.
  mlir::registerPass([] { return createLinalgMatmulTilingPass(); });
  mlir::registerPass([] { return createLinalgMatmulParallelTilingPass(); });
  mlir::registerPass([] { return createLinalgMatmulKTilingPass(); });
  mlir::registerPass([] { return createLinalgGenericTilingPass(); });
  mlir::registerPass([] { return createLinalgEpilogueTileAndFusePass(); });

  mlir::PassPipelineRegistration<>(
      "cpu-pipeline",
      "Run the full mlir_edsl CPU lowering pipeline (same sequence as "
      "MLIRLowering::addCPUPasses)",
      [](mlir::OpPassManager &pm) { buildCPUPipeline(pm); });
}

} // namespace

int main(int argc, char **argv) {
  registerCPUPasses();

  // Upstream passes referenced by the CPU pipeline (canonicalize, CSE,
  // bufferization, scf-to-openmp, vector-to-llvm, ...) so they're runnable
  // standalone via -pass-pipeline for binary search, matching what
  // buildCPUPipeline already links against.
  mlir::registerTransformsPasses();
  mlir::bufferization::registerBufferizationPasses();
  mlir::memref::registerMemRefPasses();
  mlir::vector::registerVectorPasses();
  mlir::registerSCFPasses();
  mlir::registerLinalgPasses();
  mlir::registerConvertVectorToSCFPass();
  mlir::registerConvertVectorToLLVMPass();
  mlir::registerConvertSCFToOpenMPPass();
  mlir::registerArithToLLVMConversionPass();
  mlir::registerConvertControlFlowToLLVMPass();
  mlir::registerConvertFuncToLLVMPass();
  mlir::registerConvertOpenMPToLLVMPass();
  mlir::registerFinalizeMemRefToLLVMConversionPass();
  mlir::registerLowerAffinePass();
  mlir::registerReconcileUnrealizedCastsPass();
  mlir::registerSCFToControlFlowPass();
  mlir::registerUBToLLVMConversionPass();

  mlir::DialectRegistry registry;
  registerCPUDialects(registry);

  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "MLIR-EDSL optimizer driver\n", registry));
}

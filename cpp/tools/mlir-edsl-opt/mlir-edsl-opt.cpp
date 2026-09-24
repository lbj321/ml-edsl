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
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
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
  mlir::registerPass([] { return createVectorTransposeLoweringPass(); });
  mlir::registerPass([] { return createAllocaScopeCleanupPass(); });
  mlir::registerPass(
      [] { return createVectorContractToOuterProductPass(); });
  mlir::registerPass([] { return createLinalgGenericTilingPass(); });
  // Run first inside -cpu-pipeline with their defaults. Registered separately
  // so a non-default strategy can be composed ahead of it (pass options are
  // space-separated):
  //   mlir-edsl-opt in.mlir '-linalg-matmul-blocked-distribute=mr=6 nr=16' \
  //       -cpu-pipeline
  mlir::registerPass([] { return createLinalgMatmulBlockedDistributePass(); });
  mlir::registerPass([] { return createLinalgMatmulBlockedTilePass(); });
  mlir::registerPass([] { return createLinalgMatmulBlockedPackPass(); });
  mlir::registerPass([] { return createLinalgMatmulBlockedKernelPass(); });

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
  mlir::registerConvertVectorToSCFPass();
  mlir::registerConvertVectorToLLVMPass();
  mlir::vector::registerVectorPasses();

  mlir::DialectRegistry registry;
  registerCPUDialects(registry);

  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "MLIR-EDSL optimizer driver\n", registry));
}

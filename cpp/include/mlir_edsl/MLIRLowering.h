#pragma once

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/PassManager.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace mlir_edsl {

struct LoweredModule {
  std::unique_ptr<llvm::Module> module;
  std::unique_ptr<llvm::LLVMContext> context;
};

struct GPULoweredModule {
  std::string ptxImage;       // PTX source; CUDA driver JIT-compiles at load time
  std::string kernelFuncName; // mangled entry name inside PTX
};

class MLIRLowering {
public:
  using SnapshotList = std::vector<std::pair<std::string, std::string>>;

  MLIRLowering();
  MLIRLowering(mlir::MLIRContext *sharedContext, bool captureSnapshots = false);
  ~MLIRLowering() = default;

  // Lower MLIR module to llvm::Module directly (no serialization)
  LoweredModule lowerToLLVMModule(mlir::ModuleOp module);

  // Lower MLIR module to LLVM IR string (convenience, calls lowerToLLVMModule)
  std::string lowerToLLVMIR(mlir::ModuleOp module);

  // Lower MLIR module to PTX via GPU dialect pipeline (CUDA path)
  GPULoweredModule lowerToGPUModule(mlir::ModuleOp module);

  // Get the pass manager for advanced usage
  mlir::PassManager &getPassManager() { return passManager; }

  // Move captured snapshots out (only populated when captureSnapshots=true)
  SnapshotList takeSnapshots() { return std::move(snapshots); }

  // Failure-path IR — always populated on pipeline failure, regardless of captureSnapshots
  std::string takeFailureIR() { return std::move(failureIR_); }
  bool hadFailure() const { return !failureIR_.empty(); }

private:
  std::unique_ptr<mlir::MLIRContext> context;
  mlir::PassManager passManager;
  bool snapshotsEnabled = false;
  SnapshotList snapshots;
  std::string failureIR_;

  // Helper methods
  void registerRequiredDialects();
  void registerRequiredDialects(mlir::MLIRContext *ctx);
  void setupLoweringPipeline();
  void addConversionPasses();
  bool runLoweringPipeline(mlir::ModuleOp module);

  // GPU-path helpers
  void registerGPUDialects(mlir::MLIRContext *ctx);
  void addGPUConversionPasses(mlir::PassManager &pm);
  bool runGPULoweringPipeline(mlir::ModuleOp module, mlir::PassManager &pm);
};

} // namespace mlir_edsl

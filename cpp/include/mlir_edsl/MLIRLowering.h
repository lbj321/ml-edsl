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

// One argument in a gpu.launch_func call, classified by origin.
struct GPUKernelArg {
  enum class Kind { I64, F32, InputMemRef, OutputMemRef };
  Kind kind = Kind::I64;
  int64_t i64Val = 0;         // Kind::I64  — constant scalar (step, lb, ub)
  float f32Val = 0.0f;        // Kind::F32  — constant scalar (fill value, etc.)
  int paramIdx = 0;           // Kind::InputMemRef — BlockArgument index
  std::vector<int64_t> shape; // Kind::InputMemRef or Kind::OutputMemRef
};

// One gpu.launch_func call: which kernel, how to launch it, what args it needs.
struct GPUKernelLaunch {
  std::string moduleName; // gpu.module name (unique key within one MLIR function)
  std::string funcName;   // gpu.func name inside that module
  std::string ptxImage;   // PTX source for this gpu.module (filled after NVVM phase)
  uint32_t gridX = 1, gridY = 1, gridZ = 1;
  uint32_t blockX = 1, blockY = 1, blockZ = 1;
  std::vector<GPUKernelArg> args;
};

// All kernels produced for one MLIR function (e.g. fill + matmul).
struct GPULoweredModule {
  std::vector<GPUKernelLaunch> kernels;
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

  // Shared infrastructure
  void registerRequiredDialects();
  void registerRequiredDialects(mlir::MLIRContext *ctx);
  void attachInstrumentation(mlir::PassManager &pm);
  bool runPipeline(mlir::PassManager &pm, mlir::ModuleOp module);

  // CPU pipeline
  void setupLoweringPipeline();
  void addConversionPasses();
  bool runLoweringPipeline(mlir::ModuleOp module);

  // Shared pass-sequence building blocks
  void addBufferizationPasses(mlir::PassManager &pm, bool withOutParams,
                              bool withDealloc = true);
  void addSharedFinalLLVMLoweringPasses(mlir::PassManager &pm);

  // GPU-path helpers
  void registerGPUDialects(mlir::MLIRContext *ctx);
  void addGPUPreOutliningPasses(mlir::PassManager &pm);
  void addGPUNVVMPasses(mlir::PassManager &pm);
  bool runGPULoweringPipeline(mlir::ModuleOp module, mlir::PassManager &pm);
  void analyzeKernelLaunches(mlir::ModuleOp module, GPULoweredModule &result);
};

} // namespace mlir_edsl

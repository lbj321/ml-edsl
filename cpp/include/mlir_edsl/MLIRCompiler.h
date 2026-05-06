#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/Value.h"
#include "mlir_edsl/MLIRLowering.h"

namespace mlir_edsl {

class MLIRBuilder;
class MLIRExecutor;
class MLIRGPUExecutor;
class FunctionDef;
class TypeSpec;

class MLIRCompiler {
public:
  enum class State { Building, Finalized };
  enum class OptLevel { O0, O2, O3 };
  enum class Target { CPU, GPU };

  MLIRCompiler();
  ~MLIRCompiler();

  // Non-copyable, non-movable (raw pointer aliasing between members)
  MLIRCompiler(const MLIRCompiler &) = delete;
  MLIRCompiler &operator=(const MLIRCompiler &) = delete;
  MLIRCompiler(MLIRCompiler &&) = delete;
  MLIRCompiler &operator=(MLIRCompiler &&) = delete;

  // ==================== COMPILATION (Building state only) ====================
  void compileFunction(const mlir_edsl::FunctionDef &funcDef);

  // ==================== EXECUTION ====================
  uintptr_t getFunctionPointer(const std::string &name);

  // ==================== STATE MANAGEMENT ====================
  void clear();

  State getState() const { return state; }
  bool isFinalized() const { return state == State::Finalized; }

  // ==================== INSPECTION ====================
  bool hasFunction(const std::string &name) const;
  std::vector<std::string> listFunctions() const;
  std::string getModuleIR();
  std::string getUnoptLLVMIR() const;
  std::string getOptLLVMIR() const;

  // ==================== IR SNAPSHOTS ====================
  using SnapshotList = std::vector<std::pair<std::string, std::string>>;
  const SnapshotList &getLoweringSnapshots() const { return loweringSnapshots; }

  // ==================== FAILURE IR ====================
  std::string getFailureIR() const { return failureIR_; }

  // ==================== TESTING UTILITIES ====================
  void injectTestFailure();  // Testing only: inserts a type-mismatched function

  // ==================== CONFIGURATION ====================
  void setOptimizationLevel(OptLevel level);
  void enableSnapshotCapture() { captureSnapshots = true; }
  void setTarget(Target t) { target_ = t; }
  Target getTarget() const { return target_; }

  // ==================== GPU EXECUTION ====================
  // inputs: (host_ptr, shape) per argument. output: pre-allocated host buffer.
  void executeGPUFunction(
      const std::string &name,
      const std::vector<std::pair<const void *, std::vector<int64_t>>> &inputs,
      void *output,
      const std::vector<int64_t> &outputShape,
      size_t elementSize);

private:
  State state;
  OptLevel optimizationLevel;
  Target target_ = Target::CPU;

  // ==================== OWNED INFRASTRUCTURE ====================
  std::unique_ptr<mlir::MLIRContext> mlirContext;
  std::unique_ptr<mlir::OpBuilder> opBuilder;
  mlir::OwningOpRef<mlir::ModuleOp> module;

  // ==================== FUNCTION STATE ====================
  // Declared before builder/executor: builder holds raw pointers into these,
  // so they must outlive it (members destroy in reverse declaration order).
  mlir::func::FuncOp currentFunction;
  mlir::BlockArgument currentOutParam; // set when return type is aggregate
  std::unordered_map<std::string, mlir::Value> parameterMap;
  std::unordered_map<std::string, mlir::func::FuncOp> functionTable;
  std::unordered_set<std::string> compiledFunctions;

  // ==================== OWNED COMPONENTS ====================
  // Destroyed before the data they reference (maps above, context above)
  std::unique_ptr<MLIRBuilder> builder;
  std::unique_ptr<MLIRExecutor> executor;

#ifdef MLIR_EDSL_CUDA_ENABLED
  std::unique_ptr<MLIRGPUExecutor> gpuExecutor_;
  std::unordered_map<std::string, GPULoweredModule> gpuModules_;
#endif

  // ==================== IR SNAPSHOTS ====================
  SnapshotList loweringSnapshots;
  bool captureSnapshots = false;
  std::string failureIR_;

  // ==================== INTERNAL METHODS ====================
  void ensureFinalized();
  void resetFunctionState();

  // Function building (moved from MLIRBuilder)
  void createFunction(
      const std::string &name,
      const std::vector<std::pair<std::string, mlir_edsl::TypeSpec>> &params,
      mlir::Type returnType);
  void finalizeFunction(const std::string &name, mlir::Value result);

  // Type helpers
  mlir::Type convertType(const mlir_edsl::TypeSpec &typeSpec) const;
  bool isValidParameterType(const mlir_edsl::TypeSpec &type) const;
  bool isValidReturnType(const mlir_edsl::TypeSpec &type) const;
};

} // namespace mlir_edsl

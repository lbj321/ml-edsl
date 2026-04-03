#include "mlir_edsl/MLIRExecutor.h"

#include "llvm/ExecutionEngine/Orc/LLJIT.h"
#include "llvm/ExecutionEngine/Orc/ThreadSafeModule.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Support/TargetSelect.h"

#include <stdexcept>
#include <cstdlib>

namespace mlir_edsl {

MLIRExecutor::MLIRExecutor() {
  jit = nullptr;
  initialized = false;
  optimizationLevel = OptLevel::O2;
}

void MLIRExecutor::initialize() {
  if (initialized) {
    return;
  }

  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();
  llvm::InitializeNativeTargetAsmParser();

  auto jitOrError = llvm::orc::LLJITBuilder().create();
  if (!jitOrError) {
    throw std::runtime_error("Failed to create LLJIT");
  }

  jit = std::move(jitOrError.get());
  initialized = true;
}

void MLIRExecutor::compileModule(std::unique_ptr<llvm::Module> module,
                                 std::unique_ptr<llvm::LLVMContext> context,
                                 const std::vector<std::string> &functionNames) {
  initialize();

  // Save unoptimized LLVM IR if SAVE_IR environment variable is set
  const bool saveIR = std::getenv("SAVE_IR") != nullptr;
  if (saveIR) {
    std::string filename = "ir_output/module_unopt.ll";
    std::error_code EC;
    llvm::raw_fd_ostream outFile(filename, EC);
    if (!EC) {
      module->print(outFile, nullptr);
    }
  }

  optimizeModule(module.get());

  // Save optimized LLVM IR if SAVE_IR environment variable is set
  if (saveIR) {
    std::string filename = "ir_output/module_opt.ll";
    std::error_code EC;
    llvm::raw_fd_ostream outFile(filename, EC);
    if (!EC) {
      module->print(outFile, nullptr);
    }
  }

  auto tsm = llvm::orc::ThreadSafeModule(std::move(module), std::move(context));

  auto err = jit->addIRModule(std::move(tsm));
  if (err) {
    throw std::runtime_error("Failed to add module to JIT");
  }

  // Lookup and cache function pointers for requested names
  for (const auto& name : functionNames) {
    auto symbolOrError = jit->lookup(name);
    if (!symbolOrError) {
      std::string errMsg;
      llvm::raw_string_ostream os(errMsg);
      os << symbolOrError.takeError();
      throw std::runtime_error("JIT lookup failed for '" + name + "': " + errMsg);
    }
    functionPointers[name] = (void*)symbolOrError->getValue();
  }
}

uintptr_t MLIRExecutor::getFunctionPointer(const std::string &name) {
  auto it = functionPointers.find(name);
  if (it == functionPointers.end()) {
    throw std::runtime_error("Function not compiled: " + name);
  }
  return reinterpret_cast<uintptr_t>(it->second);
}

void MLIRExecutor::setOptimizationLevel(OptLevel level) {
  optimizationLevel = level;
}

void MLIRExecutor::clear() {
  if (initialized) {
    auto jitOrError = llvm::orc::LLJITBuilder().create();
    if (!jitOrError) {
      throw std::runtime_error("Failed to recreate LLJIT");
    }
    jit = std::move(jitOrError.get());
  }
  functionPointers.clear();
}

void MLIRExecutor::optimizeModule(llvm::Module *module) {
  if (optimizationLevel == OptLevel::O0)
    return;

  llvm::OptimizationLevel llvmLevel =
      (optimizationLevel == OptLevel::O3) ? llvm::OptimizationLevel::O3
                                           : llvm::OptimizationLevel::O2;

  llvm::PassBuilder passBuilder;
  llvm::LoopAnalysisManager loopAM;
  llvm::FunctionAnalysisManager functionAM;
  llvm::CGSCCAnalysisManager cgsccAM;
  llvm::ModuleAnalysisManager moduleAM;

  passBuilder.registerModuleAnalyses(moduleAM);
  passBuilder.registerCGSCCAnalyses(cgsccAM);
  passBuilder.registerFunctionAnalyses(functionAM);
  passBuilder.registerLoopAnalyses(loopAM);
  passBuilder.crossRegisterProxies(loopAM, functionAM, cgsccAM, moduleAM);

  llvm::ModulePassManager modulePM =
      passBuilder.buildPerModuleDefaultPipeline(llvmLevel);
  modulePM.run(*module, moduleAM);
}

} // namespace mlir_edsl

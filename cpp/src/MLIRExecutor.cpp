#include "mlir_edsl/MLIRExecutor.h"

#include "llvm/ExecutionEngine/Orc/LLJIT.h"
#include "llvm/ExecutionEngine/Orc/ThreadSafeModule.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Transforms/InstCombine/InstCombine.h"
#include "llvm/Transforms/Scalar/GVN.h"
#include "llvm/Transforms/Scalar/SimplifyCFG.h"
#include "llvm/Transforms/Utils/Mem2Reg.h"

#include <stdexcept>
#include <cstdlib>

namespace mlir_edsl {

MLIRExecutor::MLIRExecutor() {
  context = std::make_unique<llvm::LLVMContext>();
  jit = nullptr;
  initialized = false;
  lastError = "";
  optimizationLevel = OptLevel::O2;
}

bool MLIRExecutor::initialize() {
  if (initialized) {
    return true;
  }

  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();
  llvm::InitializeNativeTargetAsmParser();

  auto jitOrError = llvm::orc::LLJITBuilder().create();
  if (!jitOrError) {
    lastError = "Failed to create LLJIT";
    return false;
  }

  jit = std::move(jitOrError.get());
  initialized = true;

  return true;
}

bool MLIRExecutor::compileModule(const std::string &llvmIR,
                                 const std::vector<std::string> &functionNames) {
  if (!initialize()) {
    return false;
  }

  llvm::SMDiagnostic error;
  auto buffer = llvm::MemoryBuffer::getMemBuffer(llvmIR);
  auto module = llvm::parseIR(*buffer, error, *context);

  if (!module) {
    lastError = "Failed to parse LLVM IR";
    return false;
  }

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

  auto tsm = llvm::orc::ThreadSafeModule(std::move(module),
                                         std::make_unique<llvm::LLVMContext>());

  auto err = jit->addIRModule(std::move(tsm));
  if (err) {
    lastError = "Failed to add module to JIT";
    return false;
  }

  // Lookup and cache function pointers for requested names
  for (const auto& name : functionNames) {
    auto symbolOrError = jit->lookup(name);
    if (symbolOrError) {
      functionPointers[name] = (void*)symbolOrError->getValue();
    }
  }

  return true;
}

uintptr_t MLIRExecutor::getFunctionPointer(const std::string &name) {
  auto it = functionPointers.find(name);
  if (it == functionPointers.end()) {
    lastError = "Function not compiled: " + name;
    throw std::runtime_error(lastError);
  }
  return reinterpret_cast<uintptr_t>(it->second);
}

void MLIRExecutor::setOptimizationLevel(OptLevel level) {
  optimizationLevel = level;
}

void MLIRExecutor::clear() {
  if (initialized) {
    // Create a fresh JIT instance
    auto jitOrError = llvm::orc::LLJITBuilder().create();
    if (jitOrError) {
      jit = std::move(jitOrError.get());
    } else {
      lastError = "Failed to recreate LLJIT";
    }
  }
  functionPointers.clear();
}

void MLIRExecutor::optimizeModule(llvm::Module *module) {
  if (optimizationLevel == OptLevel::O0)
    return;

  llvm::PassBuilder passBuilder;
  llvm::FunctionPassManager functionPM;
  llvm::ModulePassManager modulePM;

  if (optimizationLevel == OptLevel::O2 || optimizationLevel == OptLevel::O3) {
    functionPM.addPass(llvm::PromotePass());
    functionPM.addPass(llvm::InstCombinePass());
    functionPM.addPass(llvm::SimplifyCFGPass());

    if (optimizationLevel == OptLevel::O3) {
      functionPM.addPass(llvm::GVNPass());
    }
  }

  if (!functionPM.isEmpty()) {
    modulePM.addPass(
        llvm::createModuleToFunctionPassAdaptor(std::move(functionPM)));
  }

  llvm::LoopAnalysisManager loopAM;
  llvm::FunctionAnalysisManager functionAM;
  llvm::CGSCCAnalysisManager cgsccAM;
  llvm::ModuleAnalysisManager moduleAM;

  passBuilder.registerModuleAnalyses(moduleAM);
  passBuilder.registerCGSCCAnalyses(cgsccAM);
  passBuilder.registerFunctionAnalyses(functionAM);
  passBuilder.registerLoopAnalyses(loopAM);
  passBuilder.crossRegisterProxies(loopAM, functionAM, cgsccAM, moduleAM);

  modulePM.run(*module, moduleAM);
}

} // namespace mlir_edsl

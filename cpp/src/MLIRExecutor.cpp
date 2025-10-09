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

void *MLIRExecutor::compileFunction(const std::string &llvmIR,
                                    const std::string &funcName) {
  if (!initialize()) {
    return nullptr;
  }

  llvm::SMDiagnostic error;
  auto buffer = llvm::MemoryBuffer::getMemBuffer(llvmIR);
  auto module = llvm::parseIR(*buffer, error, *context);

  if (!module) {
    lastError = "Failed to parse LLVM IR";
    return nullptr;
  }

  optimizeModule(module.get());

  auto tsm = llvm::orc::ThreadSafeModule(std::move(module),
                                         std::make_unique<llvm::LLVMContext>());

  auto err = jit->addIRModule(std::move(tsm));
  if (err) {
    lastError = "Failed to add module to JIT";
    return nullptr;
  }

  auto symbolOrError = jit->lookup(funcName);
  if (!symbolOrError) {
    lastError = "Failed to lookup function";
    return nullptr;
  }

  return (void *)symbolOrError->getValue();
}

int32_t MLIRExecutor::callInt32Function(void *funcPtr,
                                        const std::vector<int32_t> &intArgs,
                                        const std::vector<float> &floatArgs) {
  if (!funcPtr) {
    lastError = "Null function pointer";
    return 0;
  }

  size_t totalArgs = intArgs.size() + floatArgs.size();

  if (totalArgs == 0) {
    typedef int32_t (*FuncType)();
    auto func = reinterpret_cast<FuncType>(funcPtr);
    return func();
  } else if (intArgs.size() == 1 && floatArgs.empty()) {
    typedef int32_t (*FuncType)(int32_t);
    auto func = reinterpret_cast<FuncType>(funcPtr);
    return func(intArgs[0]);
  } else if (intArgs.size() == 2 && floatArgs.empty()) {
    typedef int32_t (*FuncType)(int32_t, int32_t);
    auto func = reinterpret_cast<FuncType>(funcPtr);
    return func(intArgs[0], intArgs[1]);
  } else if (intArgs.size() == 3 && floatArgs.empty()) {
    typedef int32_t (*FuncType)(int32_t, int32_t, int32_t);
    auto func = reinterpret_cast<FuncType>(funcPtr);
    return func(intArgs[0], intArgs[1], intArgs[2]);
  } else if (intArgs.size() == 4 && floatArgs.empty()) {
    typedef int32_t (*FuncType)(int32_t, int32_t, int32_t, int32_t);
    auto func = reinterpret_cast<FuncType>(funcPtr);
    return func(intArgs[0], intArgs[1], intArgs[2], intArgs[3]);
  } else if (intArgs.size() == 1 && floatArgs.size() == 1) {
    typedef int32_t (*FuncType)(int32_t, float);
    auto func = reinterpret_cast<FuncType>(funcPtr);
    return func(intArgs[0], floatArgs[0]);
  }

  std::string error =
      "Unsupported parameter combination: " + std::to_string(intArgs.size()) +
      " int args, " + std::to_string(floatArgs.size()) + " float args";
  lastError = error;
  throw std::runtime_error(error);
}

float MLIRExecutor::callFloatFunction(void *funcPtr,
                                      const std::vector<int32_t> &intArgs,
                                      const std::vector<float> &floatArgs) {
  if (!funcPtr) {
    lastError = "Null function pointer";
    return 0.0f;
  }

  size_t totalArgs = intArgs.size() + floatArgs.size();

  if (totalArgs == 0) {
    typedef float (*FuncType)();
    auto func = reinterpret_cast<FuncType>(funcPtr);
    return func();
  } else if (intArgs.size() == 1 && floatArgs.empty()) {
    typedef float (*FuncType)(int32_t);
    auto func = reinterpret_cast<FuncType>(funcPtr);
    return func(intArgs[0]);
  } else if (intArgs.size() == 2 && floatArgs.empty()) {
    typedef float (*FuncType)(int32_t, int32_t);
    auto func = reinterpret_cast<FuncType>(funcPtr);
    return func(intArgs[0], intArgs[1]);
  } else if (intArgs.size() == 3 && floatArgs.empty()) {
    typedef float (*FuncType)(int32_t, int32_t, int32_t);
    auto func = reinterpret_cast<FuncType>(funcPtr);
    return func(intArgs[0], intArgs[1], intArgs[2]);
  } else if (intArgs.size() == 4 && floatArgs.empty()) {
    typedef float (*FuncType)(int32_t, int32_t, int32_t, int32_t);
    auto func = reinterpret_cast<FuncType>(funcPtr);
    return func(intArgs[0], intArgs[1], intArgs[2], intArgs[3]);
  } else if (floatArgs.size() == 2 && intArgs.empty()) {
    typedef float (*FuncType)(float, float);
    auto func = reinterpret_cast<FuncType>(funcPtr);
    return func(floatArgs[0], floatArgs[1]);
  } else if (intArgs.size() == 1 && floatArgs.size() == 1) {
    typedef float (*FuncType)(int32_t, float);
    auto func = reinterpret_cast<FuncType>(funcPtr);
    return func(intArgs[0], floatArgs[0]);
  }

  std::string error =
      "Unsupported parameter combination: " + std::to_string(intArgs.size()) +
      " int args, " + std::to_string(floatArgs.size()) + " float args";
  lastError = error;
  throw std::runtime_error(error);
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
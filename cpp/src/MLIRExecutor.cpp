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

  // OpenMP runtime symbols (__kmpc_fork_call etc.) are resolved lazily at
  // JIT link time via the process symbol table — no explicit load needed.

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


#ifdef MLIR_EDSL_CUDA_ENABLED

void MLIRGPUExecutor::checkCU(CUresult result, const char *msg) {
  if (result != CUDA_SUCCESS) {
    const char *errStr = nullptr;
    cuGetErrorString(result, &errStr);
    throw std::runtime_error(std::string(msg) + ": " +
                             (errStr ? errStr : "unknown CUDA error"));
  }
}

void MLIRGPUExecutor::initialize() {
  if (initialized_)
    return;
  checkCU(cuInit(0), "cuInit");
  checkCU(cuDeviceGet(&device_, 0), "cuDeviceGet");
  checkCU(cuCtxCreate(&context_, 0, device_), "cuCtxCreate");
  initialized_ = true;
}

MLIRGPUExecutor::MLIRGPUExecutor() = default;

MLIRGPUExecutor::~MLIRGPUExecutor() {
  for (auto &[name, mod] : cuModules_)
    cuModuleUnload(mod);
  if (context_)
    cuCtxDestroy(context_);
}

void MLIRGPUExecutor::loadKernel(const std::string &moduleName,
                                  const std::string &ptxImage,
                                  const std::string &funcName) {
  initialize();
  if (cuModules_.find(moduleName) == cuModules_.end()) {
    CUmodule cuMod;
    checkCU(cuModuleLoadData(&cuMod, ptxImage.c_str()), "cuModuleLoadData");
    cuModules_[moduleName] = cuMod;
  }
  CUfunction func;
  checkCU(cuModuleGetFunction(&func, cuModules_[moduleName], funcName.c_str()),
          "cuModuleGetFunction");
  kernels_[moduleName] = func;
}

void MLIRGPUExecutor::launchKernel(const std::string &moduleName,
                                    void **kernelArgs,
                                    uint32_t gridX, uint32_t gridY,
                                    uint32_t gridZ, uint32_t blockX,
                                    uint32_t blockY, uint32_t blockZ) {
  auto it = kernels_.find(moduleName);
  if (it == kernels_.end())
    throw std::runtime_error("GPU kernel not loaded: " + moduleName);

  checkCU(cuLaunchKernel(it->second, gridX, gridY, gridZ, blockX, blockY,
                         blockZ, 0, nullptr, kernelArgs, nullptr),
          "cuLaunchKernel");
  checkCU(cuCtxSynchronize(), "cuCtxSynchronize");
}

CUdeviceptr MLIRGPUExecutor::allocDevice(size_t bytes) {
  initialize();
  CUdeviceptr ptr;
  checkCU(cuMemAlloc(&ptr, bytes), "cuMemAlloc");
  return ptr;
}

void MLIRGPUExecutor::freeDevice(CUdeviceptr ptr) {
  checkCU(cuMemFree(ptr), "cuMemFree");
}

void MLIRGPUExecutor::copyH2D(CUdeviceptr dst, const void *src, size_t bytes) {
  checkCU(cuMemcpyHtoD(dst, src, bytes), "cuMemcpyHtoD");
}

void MLIRGPUExecutor::copyD2H(void *dst, CUdeviceptr src, size_t bytes) {
  checkCU(cuMemcpyDtoH(dst, src, bytes), "cuMemcpyDtoH");
}

void MLIRGPUExecutor::synchronize() {
  checkCU(cuCtxSynchronize(), "cuCtxSynchronize");
}

void MLIRGPUExecutor::clear() {
  kernels_.clear();
  for (auto &[name, mod] : cuModules_)
    cuModuleUnload(mod);
  cuModules_.clear();
}

#endif // MLIR_EDSL_CUDA_ENABLED

} // namespace mlir_edsl

#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
#include <cstdint>

#include "llvm/ExecutionEngine/Orc/LLJIT.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"

#ifdef MLIR_EDSL_CUDA_ENABLED
#include <cuda.h>
#endif

namespace mlir_edsl {

class MLIRExecutor {
   public:
    MLIRExecutor();
    ~MLIRExecutor() = default;

    // Initialize the JIT execution engine (throws on failure)
    void initialize();

    // Compile llvm::Module directly and look up the given function names (throws on failure)
    void compileModule(std::unique_ptr<llvm::Module> module,
                       std::unique_ptr<llvm::LLVMContext> context,
                       const std::vector<std::string> &functionNames);

    // Get function pointer as integer (for Python ctypes, throws if not found)
    uintptr_t getFunctionPointer(const std::string &name);

    // Reset JIT and cached function pointers
    void clear();

    bool isInitialized() const { return initialized; }

    enum class OptLevel { O0, O2, O3 };
    void setOptimizationLevel(OptLevel level);

   private:
    std::unique_ptr<llvm::orc::LLJIT> jit;
    bool initialized;

    OptLevel optimizationLevel;
    void optimizeModule(llvm::Module *module);

    // Store compiled function pointers
    std::unordered_map<std::string, void*> functionPointers;
};


#ifdef MLIR_EDSL_CUDA_ENABLED

class MLIRGPUExecutor {
public:
  MLIRGPUExecutor();
  ~MLIRGPUExecutor();

  // Load a PTX image and register the named kernel
  void loadKernel(const std::string &funcName, const std::string &ptxImage,
                  const std::string &kernelName);

  // Launch a previously loaded kernel.
  // kernelArgs: array of void* each pointing to an argument value
  void launchKernel(const std::string &funcName, void **kernelArgs,
                    uint32_t gridX, uint32_t gridY, uint32_t gridZ,
                    uint32_t blockX, uint32_t blockY, uint32_t blockZ);

  CUdeviceptr allocDevice(size_t bytes);
  void freeDevice(CUdeviceptr ptr);
  void copyH2D(CUdeviceptr dst, const void *src, size_t bytes);
  void copyD2H(void *dst, CUdeviceptr src, size_t bytes);
  void synchronize();
  void clear();

private:
  CUdevice device_ = 0;
  CUcontext context_ = nullptr;
  CUmodule cuModule_ = nullptr;
  std::unordered_map<std::string, CUfunction> kernels_;
  bool initialized_ = false;

  void initialize();
  static void checkCU(CUresult result, const char *msg);
};

#endif // MLIR_EDSL_CUDA_ENABLED

}  // namespace mlir_edsl

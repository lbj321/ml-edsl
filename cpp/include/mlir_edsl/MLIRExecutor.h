#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
#include <cstdint>

#include "llvm/ExecutionEngine/Orc/LLJIT.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"

namespace mlir_edsl {

class MLIRExecutor {
   public:
    MLIRExecutor();
    ~MLIRExecutor() = default;

    // Initialize the JIT execution engine
    bool initialize();

    // Compile LLVM IR module and look up the given function names
    bool compileModule(const std::string &llvmIR,
                       const std::vector<std::string> &functionNames);

    // Get function pointer as integer (for Python ctypes)
    uintptr_t getFunctionPointer(const std::string &name);

    // Reset JIT and cached function pointers
    void clear();

    // Utility methods
    bool isInitialized() const { return initialized; }
    std::string getLastError() const { return lastError; }

    enum class OptLevel { O0, O2, O3 };
    void setOptimizationLevel(OptLevel level);

   private:
    std::unique_ptr<llvm::orc::LLJIT> jit;
    std::unique_ptr<llvm::LLVMContext> context;
    bool initialized;
    std::string lastError;

    OptLevel optimizationLevel;
    void optimizeModule(llvm::Module *module);

    // Store compiled function pointers
    std::unordered_map<std::string, void*> functionPointers;
};

}  // namespace mlir_edsl

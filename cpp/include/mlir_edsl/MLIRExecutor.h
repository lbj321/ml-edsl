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

}  // namespace mlir_edsl

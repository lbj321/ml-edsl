#pragma once

#include <memory>
#include <string>

#include "llvm/ExecutionEngine/Orc/LLJIT.h"
#include "llvm/ExecutionEngine/Orc/ThreadSafeModule.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/TargetSelect.h"

namespace mlir_edsl {

class MLIRExecutor {
   public:
    MLIRExecutor();
    ~MLIRExecutor() = default;

    // Initialize the JIT execution engine
    bool initialize();

    // Compile LLVM IR string to executable function
    void *compileFunction(const std::string &llvmIR,
                          const std::string &funcName);

    // Execute compiled functions with different return types
    int32_t callInt32Function(void *funcPtr,
                              const std::vector<int32_t> &intArgs = {},
                              const std::vector<float> &floatArgs = {});

    float callFloatFunction(void *funcPtr,
                            const std::vector<int32_t> &intArgs = {},
                            const std::vector<float> &floatArgs = {});

    // Utility methods
    bool isInitialized() const { return initialized; }
    std::string getLastError() const { return lastError; }
    void clear(); // Clear the JIT engine

    enum class OptLevel { O0, O2, O3 };
    void setOptimizationLevel(OptLevel level);

   private:
    std::unique_ptr<llvm::orc::LLJIT> jit;
    std::unique_ptr<llvm::LLVMContext> context;
    bool initialized;
    std::string lastError;

    OptLevel optimizationLevel;
    void optimizeModule(llvm::Module *module);
};

}  // namespace mlir_edsl
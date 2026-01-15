#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include <cstdint>

#include "llvm/ExecutionEngine/Orc/LLJIT.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "ast.pb.h"

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

    // Register function signature from protobuf object
    void registerFunctionSignature(const mlir_edsl::FunctionSignature &signature);

    // Get function pointer as integer (for Python ctypes)
    uintptr_t getFunctionPointer(const std::string &name);

    // Get function signature as protobuf (returns serialized FunctionSignature)
    std::string getFunctionSignature(const std::string &name) const;

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

    // Store function signatures as protobuf objects
    std::unordered_map<std::string, FunctionSignature> signatures;

    // Store compiled function pointers
    std::unordered_map<std::string, void*> functionPointers;
};

}  // namespace mlir_edsl
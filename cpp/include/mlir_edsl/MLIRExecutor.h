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

    // Compile entire LLVM IR module (all functions at once)
    bool compileModule(const std::string &llvmIR);

    // Register function signature from protobuf object
    void registerFunctionSignature(const mlir_edsl::FunctionSignature &signature);

    // Get function pointer as integer (for Python ctypes)
    uintptr_t getFunctionPointer(const std::string &name);

    // Get function signature as protobuf (returns serialized FunctionSignature)
    std::string getFunctionSignature(const std::string &name) const;

    // JIT state management
    bool isJitEmpty() const { return functionPointers.empty(); }
    void clearJit();   // Clear JIT only, keep signatures
    void clearAll();   // Clear JIT and signatures

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

    // Store function signatures as protobuf objects
    std::unordered_map<std::string, FunctionSignature> signatures;

    // Store compiled function pointers
    std::unordered_map<std::string, void*> functionPointers;
};

}  // namespace mlir_edsl
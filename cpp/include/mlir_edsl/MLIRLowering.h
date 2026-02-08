#pragma once

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/PassManager.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include <memory>
#include <string>

namespace mlir_edsl {

struct LoweredModule {
    std::unique_ptr<llvm::Module> module;
    std::unique_ptr<llvm::LLVMContext> context;
};

class MLIRLowering {
public:
    MLIRLowering();
    MLIRLowering(mlir::MLIRContext* sharedContext);
    ~MLIRLowering() = default;

    // Lower MLIR module to llvm::Module directly (no serialization)
    LoweredModule lowerToLLVMModule(mlir::ModuleOp module);

    // Lower MLIR module to LLVM IR string (convenience, calls lowerToLLVMModule)
    std::string lowerToLLVMIR(mlir::ModuleOp module);

    // Get the pass manager for advanced usage
    mlir::PassManager& getPassManager() { return passManager; }

private:
    std::unique_ptr<mlir::MLIRContext> context;
    mlir::PassManager passManager;

    // Helper methods
    void registerRequiredDialects();
    void registerRequiredDialects(mlir::MLIRContext* ctx);
    void setupLoweringPipeline();
    void addConversionPasses();
    bool runLoweringPipeline(mlir::ModuleOp module);
};

} // namespace mlir_edsl

#pragma once

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include <memory>
#include <string>

namespace mlir_edsl {

class MLIRLowering {
public:
    MLIRLowering();
    MLIRLowering(mlir::MLIRContext* sharedContext);
    ~MLIRLowering() = default;

    // Setup the lowering pipeline with necessary passes
    void setupLoweringPipeline();
    
    // Lower MLIR module to LLVM IR string
    std::string lowerToLLVMIR(mlir::ModuleOp module);
    
    // Get the pass manager for advanced usage
    mlir::PassManager& getPassManager() { return passManager; }

private:
    std::unique_ptr<mlir::MLIRContext> context;
    mlir::PassManager passManager;
    
    // Helper methods
    void registerRequiredDialects();
    void registerRequiredDialects(mlir::MLIRContext* ctx);
    void addConversionPasses();
    bool runLoweringPipeline(mlir::ModuleOp module);
};

} // namespace mlir_edsl
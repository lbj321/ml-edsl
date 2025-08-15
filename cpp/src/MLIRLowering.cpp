#include "mlir_edsl/MLIRLowering.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "llvm/Support/raw_ostream.h"
#include <iostream>

namespace mlir_edsl {

MLIRLowering::MLIRLowering() 
    : context(std::make_unique<mlir::MLIRContext>()), 
      passManager(context.get()) {

    registerRequiredDialects();
    setupLoweringPipeline();
}

MLIRLowering::MLIRLowering(mlir::MLIRContext* sharedContext) 
    : context(nullptr), 
      passManager(sharedContext) {

    registerRequiredDialects(sharedContext);
    setupLoweringPipeline();
}

void MLIRLowering::registerRequiredDialects() {
    registerRequiredDialects(context.get());
}

void MLIRLowering::registerRequiredDialects(mlir::MLIRContext* context) {
    context->getOrLoadDialect<mlir::arith::ArithDialect>();
    context->getOrLoadDialect<mlir::func::FuncDialect>();
    context->getOrLoadDialect<mlir::LLVM::LLVMDialect>();
    
    // Register LLVM translation interfaces
    mlir::registerLLVMDialectTranslation(*context);
    mlir::registerBuiltinDialectTranslation(*context);
}

void MLIRLowering::setupLoweringPipeline() {
    passManager.enableTiming();
    addConversionPasses();
}

void MLIRLowering::addConversionPasses() {
    passManager.addPass(mlir::createArithToLLVMConversionPass());
    passManager.addPass(mlir::createConvertFuncToLLVMPass());
}

bool MLIRLowering::runLoweringPipeline(mlir::ModuleOp module) {
    if (mlir::failed(passManager.run(module))) {
        std::cerr << "Failed to run lowering pipeline\n";
        return false;
    }
    return true;
}

std::string MLIRLowering::lowerToLLVMIR(mlir::ModuleOp module) {
    mlir::ModuleOp clonedModule = module.clone();
    
    if (!runLoweringPipeline(clonedModule)) {
        return "ERROR: Lowering pipeline failed";
    }
    
    llvm::LLVMContext llvmContext;
    auto llvmModule = mlir::translateModuleToLLVMIR(clonedModule, llvmContext);

    if (!llvmModule) {
        return "ERROR: Translation to LLVM IR failed";
    }

    std::string result;
    llvm::raw_string_ostream stream(result);
    llvmModule->print(stream, nullptr);
    return result;
}

} // namespace mlir_edsl
#include "mlir_edsl/MLIRLowering.h"

#include <iostream>

#include "mlir/Pass/Pass.h"

#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir_edsl {

MLIRLowering::MLIRLowering()
    : context(std::make_unique<mlir::MLIRContext>()),
      passManager(context.get()) {
  registerRequiredDialects();
  setupLoweringPipeline();
}

MLIRLowering::MLIRLowering(mlir::MLIRContext *sharedContext)
    : context(nullptr), passManager(sharedContext) {
  registerRequiredDialects(sharedContext);
  setupLoweringPipeline();
}

void MLIRLowering::registerRequiredDialects() {
  registerRequiredDialects(context.get());
}

void MLIRLowering::registerRequiredDialects(mlir::MLIRContext *context) {
  context->getOrLoadDialect<mlir::arith::ArithDialect>();
  context->getOrLoadDialect<mlir::func::FuncDialect>();
  context->getOrLoadDialect<mlir::scf::SCFDialect>();
  context->getOrLoadDialect<mlir::cf::ControlFlowDialect>();
  context->getOrLoadDialect<mlir::LLVM::LLVMDialect>();

  // Register LLVM translation interfaces
  mlir::registerLLVMDialectTranslation(*context);
  mlir::registerBuiltinDialectTranslation(*context);
}

void MLIRLowering::setupLoweringPipeline() {
  passManager.enableTiming();
  passManager.enableVerifier(true);
}

void MLIRLowering::addConversionPasses() {
  // First lower SCF to ControlFlow dialect
  passManager.addPass(mlir::createSCFToControlFlowPass());

  // Then lower everything to LLVM
  passManager.addPass(mlir::createArithToLLVMConversionPass());
  passManager.addPass(mlir::createConvertControlFlowToLLVMPass());
  passManager.addPass(mlir::createConvertFuncToLLVMPass());
  passManager.addPass(mlir::createReconcileUnrealizedCastsPass());
}

bool MLIRLowering::runLoweringPipeline(mlir::ModuleOp module) {
  const bool debug = std::getenv("MLIR_DEBUG") != nullptr;

  if (debug) {
    std::cerr << "=== BEFORE LOWERING ===\n";
    module.print(llvm::errs());
    std::cerr << "\n========================\n";
  }

  passManager.clear();
  addConversionPasses();

  if (mlir::failed(passManager.run(module))) {
    std::cerr << "Failed to run lowering pipeline\n";
    if (debug) {
      std::cerr << "=== MODULE AFTER FAILED LOWERING ===\n";
      module.print(llvm::errs());
      std::cerr << "\n==================================\n";
    }
    return false;
  }

  if (debug) {
    std::cerr << "=== AFTER SUCCESSFUL LOWERING ===\n";
    module.print(llvm::errs());
    std::cerr << "\n===============================\n";
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
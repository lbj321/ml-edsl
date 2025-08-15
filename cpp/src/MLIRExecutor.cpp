#include "mlir_edsl/MLIRExecutor.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/ExecutionEngine/Orc/LLJIT.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/ExecutionEngine/Orc/ThreadSafeModule.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Utils/Mem2Reg.h"
#include "llvm/Transforms/InstCombine/InstCombine.h"
#include "llvm/Transforms/Scalar/SimplifyCFG.h"
#include "llvm/Transforms/Scalar/GVN.h"

namespace mlir_edsl {

MLIRExecutor::MLIRExecutor() {
    context = std::make_unique<llvm::LLVMContext>();
    jit = nullptr;
    initialized = false;
    lastError = "";
    optimizationLevel = OptLevel::O2;
}

bool MLIRExecutor::initialize() {
    if (initialized) {
        return true;
    }

    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmPrinter();
    llvm::InitializeNativeTargetAsmParser();

    auto jitOrError = llvm::orc::LLJITBuilder().create();
    if (!jitOrError) {
        lastError = "Failed to create LLJIT";
        return false;
    }

    jit = std::move(jitOrError.get());
    initialized = true;

    return true;
}

void* MLIRExecutor::compileFunction(const std::string& llvmIR, const std::string& funcName) {
    if (!initialize()) {
        return nullptr;
    }

    llvm::SMDiagnostic error;
    auto buffer = llvm::MemoryBuffer::getMemBuffer(llvmIR);
    auto module = llvm::parseIR(*buffer, error, *context);

    if (!module) {
        lastError = "Failed to parse LLVM IR";
        return nullptr;
    }

    optimizeModule(module.get());

    auto tsm = llvm::orc::ThreadSafeModule(std::move(module),
                                           std::make_unique<llvm::LLVMContext>());
            
    auto err = jit->addIRModule(std::move(tsm));
    if (err) {
        lastError = "Failed to add module to JIT";
        return nullptr;
    }

    auto symbolOrError = jit->lookup(funcName);
    if (!symbolOrError) {
        lastError = "Failed to lookup function";
        return nullptr;
    }

    return (void*)symbolOrError->getValue();
}

int32_t MLIRExecutor::callInt32Function(void* funcPtr) {
    if (!funcPtr) {
        lastError = "Null function pointer";
        return 0;
    }

    typedef int32_t (*FuncType)();
    auto func = reinterpret_cast<FuncType>(funcPtr);
    return func();
}

float MLIRExecutor::callFloatFunction(void* funcPtr) {
    if (!funcPtr) {
        lastError = "Null function pointer";
        return 0.0f;
    }

    typedef float (*FuncType)();
    auto func = reinterpret_cast<FuncType>(funcPtr);
    return func();
}

void MLIRExecutor::setOptimizationLevel(OptLevel level) {
    optimizationLevel = level;
}

void MLIRExecutor::optimizeModule(llvm::Module* module) {
    if (optimizationLevel == OptLevel::O0)
        return;

    llvm::PassBuilder passBuilder;
    llvm::FunctionPassManager functionPM;
    llvm::ModulePassManager modulePM;

    if (optimizationLevel == OptLevel::O2 || optimizationLevel == OptLevel::O3) {
        functionPM.addPass(llvm::PromotePass());
        functionPM.addPass(llvm::InstCombinePass());
        functionPM.addPass(llvm::SimplifyCFGPass());

        if (optimizationLevel == OptLevel::O3) {
            functionPM.addPass(llvm::GVNPass());
        }
    }

    if (!functionPM.isEmpty()) {
        modulePM.addPass(llvm::createModuleToFunctionPassAdaptor(std::move(functionPM)));
    }

    llvm::LoopAnalysisManager loopAM;
    llvm::FunctionAnalysisManager functionAM;
    llvm::CGSCCAnalysisManager cgsccAM;
    llvm::ModuleAnalysisManager moduleAM;

    passBuilder.registerModuleAnalyses(moduleAM);
    passBuilder.registerCGSCCAnalyses(cgsccAM);
    passBuilder.registerFunctionAnalyses(functionAM);
    passBuilder.registerLoopAnalyses(loopAM);
    passBuilder.crossRegisterProxies(loopAM, functionAM, cgsccAM, moduleAM);

    modulePM.run(*module, moduleAM);
}

} // namespace mlir_edsl
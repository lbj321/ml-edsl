# MLIRDialects.cmake - MLIR dialect library configuration

# Core MLIR libraries used across the project
set(MLIR_CORE_LIBS
    MLIRIR
    MLIRSupport
    MLIRParser
    MLIRTransforms
    MLIRPass
)

# Arithmetic dialect libraries
set(MLIR_ARITH_LIBS
    MLIRArithDialect
    MLIRArithUtils
    MLIRArithToLLVM
)

# Function dialect libraries  
set(MLIR_FUNC_LIBS
    MLIRFuncDialect
    MLIRFuncToLLVM
)

# LLVM dialect and translation
set(MLIR_LLVM_LIBS
    MLIRLLVMDialect
    MLIRTargetLLVMIRExport
    MLIRLLVMToLLVMIRTranslation
    MLIRBuiltinToLLVMIRTranslation
)

# Control flow dialect (for Phase 6)
set(MLIR_SCF_LIBS
    MLIRSCFDialect
    MLIRSCFToControlFlow
    MLIRControlFlowDialect
)

# Memory and tensor dialects (for Phase 7)
set(MLIR_MEMORY_LIBS
    MLIRMemRefDialect
    MLIRTensorDialect
    MLIRLinalgDialect
    MLIRBufferizationDialect
)

# Combine all current libraries
set(MLIR_CURRENT_LIBS
    ${MLIR_CORE_LIBS}
    ${MLIR_ARITH_LIBS}
    ${MLIR_FUNC_LIBS}
    ${MLIR_LLVM_LIBS}
    ${MLIR_SCF_LIBS}
    MLIRReconcileUnrealizedCasts
)

# Function to add MLIR libraries to target
function(target_link_mlir_libraries target)
    target_link_libraries(${target} PRIVATE
        ${MLIR_CURRENT_LIBS}
        ${LLVM_LIBS}
    )
endfunction()

# Function to add MLIR include directories
function(target_include_mlir_directories target)
    target_include_directories(${target} PRIVATE
        ${LLVM_INCLUDE_DIRS}
        ${MLIR_INCLUDE_DIRS}
    )
endfunction()
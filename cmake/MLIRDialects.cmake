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
    MLIRSCFTransforms
    MLIRControlFlowDialect
)

# Memory and tensor dialects (for Phase 7)
# Linalg transforms (for Phase 8.2 - provides createConvertLinalgToLoopsPass)
set(MLIR_MEMORY_LIBS
    MLIRMemRefDialect
    MLIRMemRefToLLVM
    MLIRTensorDialect
    MLIRLinalgDialect
    MLIRLinalgTransforms
    MLIRBufferizationDialect
    MLIRBufferizationTransforms
    MLIRTensorTransforms
    MLIRArithTransforms
)

# Vector dialect and lowering (Phase 9 - vectorization)
set(MLIR_VECTOR_LIBS
    MLIRVectorDialect
    MLIRVectorTransforms
    MLIRVectorToLLVMPass
    MLIRVectorToSCF
    MLIRUBToLLVM
)

# Combine all current libraries
set(MLIR_CURRENT_LIBS
    ${MLIR_CORE_LIBS}
    ${MLIR_ARITH_LIBS}
    ${MLIR_FUNC_LIBS}
    ${MLIR_LLVM_LIBS}
    ${MLIR_SCF_LIBS}
    ${MLIR_MEMORY_LIBS}
    ${MLIR_VECTOR_LIBS}
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
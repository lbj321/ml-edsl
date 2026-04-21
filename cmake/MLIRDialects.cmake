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
    MLIRArithValueBoundsOpInterfaceImpl
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
    MLIRSCFToOpenMP
    MLIRSCFTransforms
    MLIRControlFlowDialect
    MLIROpenMPDialect
    MLIROpenMPToLLVM
)

# Memory and tensor dialects (for Phase 7)
# Linalg transforms (for Phase 8.2 - provides createConvertLinalgToLoopsPass)
set(MLIR_MEMORY_LIBS
    MLIRMemRefDialect
    MLIRMemRefToLLVM
    MLIRMemRefTransforms
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
    MLIRVectorToLLVM
    MLIRVectorToLLVMPass
    MLIRVectorToSCF
    MLIRUBToLLVM
)

# GPU/CUDA dialect libraries (optional)
option(MLIR_EDSL_CUDA "Enable CUDA GPU execution backend" OFF)

if(MLIR_EDSL_CUDA)
    find_package(CUDAToolkit REQUIRED)

    set(MLIR_GPU_LIBS
        MLIRComplexToLLVM
        MLIRControlFlowToLLVM
        MLIRIndexToLLVM
        MLIRGPUDialect
        MLIRGPUToGPURuntimeTransforms
        MLIRGPUTransforms
        MLIRGPUToNVVMTransforms
        MLIRMathToLLVM
        MLIRNVVMDialect
        MLIRNVVMToLLVM
        MLIRGPUToLLVMIRTranslation
        MLIRNVVMToLLVMIRTranslation
        MLIRSCFToGPU
    )

    llvm_map_components_to_libnames(LLVM_NVPTX_LIBS
        NVPTXCodeGen NVPTXDesc NVPTXInfo
    )

    list(APPEND LLVM_LIBS ${LLVM_NVPTX_LIBS})

    add_compile_definitions(MLIR_EDSL_CUDA_ENABLED)
endif()

# Combine all current libraries
set(MLIR_CURRENT_LIBS
    ${MLIR_CORE_LIBS}
    ${MLIR_ARITH_LIBS}
    ${MLIR_FUNC_LIBS}
    ${MLIR_LLVM_LIBS}
    ${MLIR_SCF_LIBS}
    ${MLIR_MEMORY_LIBS}
    ${MLIR_VECTOR_LIBS}
    MLIRAffineToStandard
    MLIRReconcileUnrealizedCasts
)

if(MLIR_EDSL_CUDA)
    list(APPEND MLIR_CURRENT_LIBS ${MLIR_GPU_LIBS})
endif()

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
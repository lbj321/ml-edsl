# FindMLIR.cmake - Find and configure MLIR installation

if(NOT LLVM_DIR OR NOT MLIR_DIR)
    if(NOT DEFINED ENV{LLVM_DIR} OR NOT DEFINED ENV{MLIR_DIR})
        message(FATAL_ERROR 
            "LLVM_DIR and MLIR_DIR must be set. Example:\n"
            "  export LLVM_DIR=/path/to/llvm/lib/cmake/llvm\n"
            "  export MLIR_DIR=/path/to/llvm/lib/cmake/mlir\n"
            "Or pass via cmake: -DLLVM_DIR=... -DMLIR_DIR=...")
    endif()
    
    set(LLVM_DIR "$ENV{LLVM_DIR}")
    set(MLIR_DIR "$ENV{MLIR_DIR}")
endif()

# Add to CMAKE_PREFIX_PATH
list(APPEND CMAKE_PREFIX_PATH "${LLVM_DIR}" "${MLIR_DIR}")

# Find packages
find_package(LLVM REQUIRED CONFIG)
find_package(MLIR REQUIRED CONFIG)

if(LLVM_FOUND AND MLIR_FOUND)
    message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
    message(STATUS "LLVM includes: ${LLVM_INCLUDE_DIRS}")
    message(STATUS "Found MLIR includes: ${MLIR_INCLUDE_DIRS}")
    
    # Add MLIR definitions
    add_definitions(${MLIR_DEFINITIONS})
    
    # Set up LLVM components
    llvm_map_components_to_libnames(LLVM_LIBS
        core
        support
        irreader
        asmparser
        asmprinter
        bitreader
        bitwriter
        passes
        target
        native
        orcjit
        executionengine
    )
    
    set(MLIR_FOUND TRUE)
else()
    message(FATAL_ERROR "LLVM and MLIR are required but not found")
endif()
# C++ Components CMakeLists.txt

# Find pybind11 early since core library now needs it
find_package(pybind11 REQUIRED)

# MLIR generates __kmpc_* calls (LLVM OMP ABI); libgomp uses GOMP_* and does
# not provide them. MLIRExecutor::initialize() opens libomp with RTLD_GLOBAL
# so the LLJIT symbol resolver can find __kmpc_* at JIT link time. Linking
# here just makes the path available as a compile definition — it is not
# sufficient on its own because shared library deps load with RTLD_LOCAL.
find_library(LIBOMP_PATH NAMES omp
    PATHS
        ${LLVM_LIBRARY_DIR}
        /usr/lib/x86_64-linux-gnu
        /usr/local/lib
        $ENV{CONDA_PREFIX}/lib
        $ENV{HOME}/anaconda3/lib
        $ENV{HOME}/miniconda3/lib
)
if(NOT LIBOMP_PATH)
    message(WARNING "libomp not found — OMP parallel execution will abort at runtime")
endif()

# memrefCopy and similar MLIR runtime helpers live in libmlir_c_runner_utils.
# MLIRExecutor::initialize() opens it with RTLD_GLOBAL so the LLJIT resolver
# can find memrefCopy at JIT link time (same pattern as libomp above).
find_library(MLIR_C_RUNNER_UTILS_PATH NAMES mlir_c_runner_utils
    PATHS
        ${LLVM_LIBRARY_DIR}
        /usr/lib/x86_64-linux-gnu
        /usr/local/lib
)
if(NOT MLIR_C_RUNNER_UTILS_PATH)
    message(WARNING "libmlir_c_runner_utils not found — memrefCopy will be missing at JIT link time")
endif()

# Core MLIR components library
add_library(mlir_edsl_core STATIC
    src/MLIRBuilder.cpp
    src/MLIRLowering.cpp
    src/MLIRLoweringPasses.cpp
    src/builders/ArithBuilder.cpp
    src/builders/SCFBuilder.cpp
    src/builders/MemRefBuilder.cpp
    src/builders/TensorBuilder.cpp
    src/builders/LinalgBuilder.cpp
)

# MLIRLowering subclasses mlir::PassInstrumentation which requires -fno-rtti
# to match LLVM's build. Other files use pybind11 which requires RTTI.
if(NOT LLVM_ENABLE_RTTI)
    set_source_files_properties(src/MLIRLowering.cpp
        PROPERTIES COMPILE_OPTIONS "-fno-rtti")
    set_source_files_properties(src/MLIRLoweringPasses.cpp
        PROPERTIES COMPILE_OPTIONS "-fno-rtti")
    if(MLIR_EDSL_CUDA)
        set_source_files_properties(src/MLIRExecutor.cpp
            PROPERTIES COMPILE_OPTIONS "-fno-rtti")
    endif()
endif()

target_include_directories(mlir_edsl_core PUBLIC
    include
)

target_include_mlir_directories(mlir_edsl_core)
target_link_mlir_libraries(mlir_edsl_core)

# Compile Protocol Buffers schemas
compile_protobuf_schemas(mlir_edsl_core
    SCHEMAS ${CMAKE_CURRENT_SOURCE_DIR}/schemas/ast.proto
    OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR}/generated
)

# Link pybind11 to core library since MLIRBuilder now uses py::object
target_link_libraries(mlir_edsl_core PUBLIC
    pybind11::pybind11
)

# JIT Executor and Compiler component
add_library(mlir_edsl_executor STATIC
    src/MLIRExecutor.cpp
    src/MLIRCompiler.cpp
)

target_include_directories(mlir_edsl_executor PUBLIC
    include
    ${CMAKE_CURRENT_BINARY_DIR}/generated
)

target_include_mlir_directories(mlir_edsl_executor)
target_link_mlir_libraries(mlir_edsl_executor)

target_link_libraries(mlir_edsl_executor PUBLIC
    mlir_edsl_core
)

if(LIBOMP_PATH)
    target_link_libraries(mlir_edsl_executor PUBLIC ${LIBOMP_PATH})
    target_compile_definitions(mlir_edsl_executor PRIVATE
        MLIR_EDSL_LIBOMP_PATH="${LIBOMP_PATH}")
endif()

if(MLIR_C_RUNNER_UTILS_PATH)
    target_compile_definitions(mlir_edsl_executor PRIVATE
        MLIR_EDSL_C_RUNNER_UTILS_PATH="${MLIR_C_RUNNER_UTILS_PATH}")
endif()

if(MLIR_EDSL_CUDA)
    target_link_libraries(mlir_edsl_executor PUBLIC CUDA::cuda_driver)
    target_include_directories(mlir_edsl_executor PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
endif()

# Python bindings module
add_library(_mlir_backend MODULE
    src/python_bindings.cpp
)

target_include_directories(_mlir_backend PRIVATE
    include
    ${CMAKE_CURRENT_BINARY_DIR}/generated
)

target_include_mlir_directories(_mlir_backend)

target_link_libraries(_mlir_backend PRIVATE
    pybind11::module
    mlir_edsl_core
    mlir_edsl_executor
)

# Set Python module properties
set_target_properties(_mlir_backend PROPERTIES
    PREFIX ""
    OUTPUT_NAME "_mlir_backend"
)
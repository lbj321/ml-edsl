# C++ Components CMakeLists.txt

# Core MLIR components library
add_library(mlir_edsl_core STATIC
    src/MLIRBuilder.cpp
    src/MLIRLowering.cpp
)

target_include_directories(mlir_edsl_core PUBLIC
    include
)

target_include_mlir_directories(mlir_edsl_core)
target_link_mlir_libraries(mlir_edsl_core)

# JIT Executor component
add_library(mlir_edsl_executor STATIC
    src/MLIRExecutor.cpp
)

target_include_directories(mlir_edsl_executor PUBLIC
    include
)

target_include_mlir_directories(mlir_edsl_executor)
target_link_mlir_libraries(mlir_edsl_executor)

# Python bindings module
find_package(pybind11 REQUIRED)

add_library(_mlir_backend MODULE
    src/python_bindings.cpp
)

target_include_directories(_mlir_backend PRIVATE
    include
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
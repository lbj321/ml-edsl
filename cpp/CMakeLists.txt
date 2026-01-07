# C++ Components CMakeLists.txt

# Find pybind11 early since core library now needs it
find_package(pybind11 REQUIRED)

# Core MLIR components library
add_library(mlir_edsl_core STATIC
    src/MLIRBuilder.cpp
    src/MLIRLowering.cpp
    src/builders/ArithBuilder.cpp
    src/builders/SCFBuilder.cpp
    src/builders/MemRefBuilder.cpp
)

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

# JIT Executor component
add_library(mlir_edsl_executor STATIC
    src/MLIRExecutor.cpp
)

target_include_directories(mlir_edsl_executor PUBLIC
    include
    ${CMAKE_CURRENT_BINARY_DIR}/generated
)

target_include_mlir_directories(mlir_edsl_executor)
target_link_mlir_libraries(mlir_edsl_executor)

# Executor now needs protobuf for FunctionSignature
target_link_libraries(mlir_edsl_executor PUBLIC
    mlir_edsl_core
    protobuf::libprotobuf
)

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
    pybind11_native_proto_caster
    protobuf::libprotobuf
)

# Set Python module properties
set_target_properties(_mlir_backend PROPERTIES
    PREFIX ""
    OUTPUT_NAME "_mlir_backend"
)
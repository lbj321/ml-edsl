cmake_minimum_required(VERSION 3.13)
project(mlir_edsl)

# Add cmake modules to path
list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)

# Include modular configuration
include(CompilerFlags)
include(FindMLIR)
include(MLIRDialects)

# Fetch pybind11_protobuf first - it will provide protobuf
include(Pybind11Protobuf)

# Now compile protobuf schemas using the fetched protobuf
include(ProtobufSchemas)

# Build C++ components
add_subdirectory(cpp)

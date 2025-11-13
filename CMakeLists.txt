cmake_minimum_required(VERSION 3.13)
project(mlir_edsl)

# Add cmake modules to path
list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)

# Include modular configuration
include(CompilerFlags)
include(FindMLIR)
include(MLIRDialects)
include(ProtobufSchemas)

# Find Protocol Buffers
find_package(Protobuf REQUIRED)

# Build C++ components
add_subdirectory(cpp)

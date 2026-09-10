# Standalone mlir-opt clone + LowerUnpackDirectPass, built against an
# out-of-tree LLVM/MLIR build (default: ~/dev/llvm-project/build, override
# with -DMLIR_DIR=<build>/lib/cmake/mlir -DLLVM_DIR=<build>/lib/cmake/llvm).
# Not part of the main project's build - this is scratch tooling for
# experiments/ab-panel-pack-matmul only.
cmake_minimum_required(VERSION 3.20)
project(ab-panel-pack-matmul-standalone-opt LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT DEFINED MLIR_DIR)
  set(MLIR_DIR "$ENV{HOME}/dev/llvm-project/build/lib/cmake/mlir")
endif()
if(NOT DEFINED LLVM_DIR)
  set(LLVM_DIR "$ENV{HOME}/dev/llvm-project/build/lib/cmake/llvm")
endif()

find_package(MLIR REQUIRED CONFIG)

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
add_definitions(${LLVM_DEFINITIONS})

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
get_property(extension_libs GLOBAL PROPERTY MLIR_EXTENSION_LIBS)

add_llvm_executable(standalone-opt
  standalone-opt.cpp
  LowerUnpackDirectPass.cpp
)

target_link_libraries(standalone-opt PRIVATE
  ${dialect_libs}
  ${conversion_libs}
  ${extension_libs}
  MLIROptLib
  MLIRPass
  MLIRTransforms
  MLIRLinalgDialect
  MLIRLinalgTransforms
  MLIRLinalgUtils
  MLIRBufferizationDialect
  MLIRTensorDialect
  MLIRFuncDialect
)

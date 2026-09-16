add_executable(mlir-edsl-opt mlir-edsl-opt.cpp)

target_include_directories(mlir-edsl-opt PRIVATE
    ${CMAKE_SOURCE_DIR}/cpp/include
)

target_include_mlir_directories(mlir-edsl-opt)
target_link_mlir_libraries(mlir-edsl-opt)

target_link_libraries(mlir-edsl-opt PRIVATE
    mlir_edsl_core
    MLIROptLib
)

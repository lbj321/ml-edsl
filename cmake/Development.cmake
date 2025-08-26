# Development.cmake - Development and testing utilities

# Option to enable development warnings
option(MLIR_EDSL_ENABLE_WARNINGS "Enable additional compiler warnings" OFF)

# Option to build with sanitizers (for debugging)
option(MLIR_EDSL_ENABLE_SANITIZERS "Enable AddressSanitizer and UBSan" OFF)

# Option for Phase 6 SCF dialect support (future)
option(MLIR_EDSL_ENABLE_SCF "Enable Structured Control Flow dialect support" OFF)

# Development mode configuration
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(MLIR_EDSL_ENABLE_WARNINGS ON CACHE BOOL "Enable warnings in debug mode" FORCE)
endif()

# Sanitizers for debugging (Linux/macOS only)
if(MLIR_EDSL_ENABLE_SANITIZERS AND NOT WIN32)
    set(SANITIZER_FLAGS "-fsanitize=address -fsanitize=undefined -fno-omit-frame-pointer")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${SANITIZER_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${SANITIZER_FLAGS}")
    message(STATUS "Enabled AddressSanitizer and UBSanitizer")
endif()

# SCF dialect support (for Phase 6)
if(MLIR_EDSL_ENABLE_SCF)
    add_compile_definitions(MLIR_EDSL_ENABLE_SCF)
    message(STATUS "Structured Control Flow dialect enabled")
endif()

# Function to create a test executable
function(add_mlir_edsl_test test_name)
    add_executable(${test_name} ${ARGN})
    target_include_directories(${test_name} PRIVATE cpp/include)
    target_include_mlir_directories(${test_name})
    target_link_libraries(${test_name} PRIVATE mlir_edsl_core mlir_edsl_executor)
    
    # Add to CTest
    add_test(NAME ${test_name} COMMAND ${test_name})
endfunction()
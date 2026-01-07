# Pybind11Protobuf.cmake - Fetch and configure pybind11_protobuf

include(FetchContent)

# Disable all testing for fetched content
set(BUILD_TESTING OFF CACHE BOOL "Disable testing" FORCE)

# Let pybind11_protobuf fetch and manage protobuf version
# This ensures version compatibility between pybind11_protobuf and protobuf
set(protobuf_INSTALL OFF CACHE BOOL "Don't install protobuf" FORCE)
set(protobuf_BUILD_TESTS OFF CACHE BOOL "Don't build protobuf tests" FORCE)
set(protobuf_BUILD_PROTOC_BINARIES ON CACHE BOOL "Build protoc compiler" FORCE)

# Disable pybind11_protobuf tests and examples
set(pybind11_protobuf_INSTALL OFF CACHE BOOL "Don't install pybind11_protobuf" FORCE)

# Fetch pybind11_protobuf from GitHub
# It will automatically fetch a compatible protobuf version
FetchContent_Declare(
  pybind11_protobuf
  GIT_REPOSITORY https://github.com/pybind/pybind11_protobuf.git
  GIT_TAG main
)

# Make available - this will also fetch protobuf as a dependency
FetchContent_MakeAvailable(pybind11_protobuf)

message(STATUS "pybind11_protobuf fetched with compatible protobuf version")

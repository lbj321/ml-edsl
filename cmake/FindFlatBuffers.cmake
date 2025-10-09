# FindFlatBuffers.cmake - Find and configure FlatBuffers installation

# Find the FlatBuffers include directory
find_path(FlatBuffers_INCLUDE_DIR
    NAMES flatbuffers/flatbuffers.h
    PATHS
        /usr/include
        /usr/local/include
        /opt/local/include
    DOC "FlatBuffers include directory"
)

# Find the FlatBuffers library
find_library(FlatBuffers_LIBRARY
    NAMES flatbuffers
    PATHS
        /usr/lib
        /usr/local/lib
        /opt/local/lib
    DOC "FlatBuffers library"
)

# Find the flatc compiler
find_program(FlatBuffers_COMPILER
    NAMES flatc
    PATHS
        /usr/bin
        /usr/local/bin
        /opt/local/bin
    DOC "FlatBuffers compiler"
)

# Handle standard find_package arguments
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(FlatBuffers
    REQUIRED_VARS FlatBuffers_INCLUDE_DIR FlatBuffers_LIBRARY FlatBuffers_COMPILER
)

# Create imported targets if found
if(FlatBuffers_FOUND)
    # Create the main library target
    if(NOT TARGET flatbuffers::flatbuffers)
        add_library(flatbuffers::flatbuffers UNKNOWN IMPORTED)
        set_target_properties(flatbuffers::flatbuffers PROPERTIES
            IMPORTED_LOCATION "${FlatBuffers_LIBRARY}"
            INTERFACE_INCLUDE_DIRECTORIES "${FlatBuffers_INCLUDE_DIR}"
        )
    endif()

    # Create the compiler target
    if(NOT TARGET flatbuffers::flatc)
        add_executable(flatbuffers::flatc IMPORTED)
        set_target_properties(flatbuffers::flatc PROPERTIES
            IMPORTED_LOCATION "${FlatBuffers_COMPILER}"
        )
    endif()

    # Set legacy variables for compatibility
    set(FlatBuffers_INCLUDE_DIRS ${FlatBuffers_INCLUDE_DIR})
    set(FlatBuffers_LIBRARIES ${FlatBuffers_LIBRARY})

    message(STATUS "Found FlatBuffers: ${FlatBuffers_LIBRARY}")
    message(STATUS "FlatBuffers include dir: ${FlatBuffers_INCLUDE_DIR}")
    message(STATUS "FlatBuffers compiler: ${FlatBuffers_COMPILER}")
endif()

# Mark variables as advanced
mark_as_advanced(
    FlatBuffers_INCLUDE_DIR
    FlatBuffers_LIBRARY
    FlatBuffers_COMPILER
)
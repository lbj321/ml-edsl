# ProtobufSchemas.cmake - Compile .proto schema files to C++ and Python

# Function to compile Protocol Buffers schemas
function(compile_protobuf_schemas target_name)
    cmake_parse_arguments(PROTO "" "OUTPUT_DIR" "SCHEMAS" ${ARGN})

    # Set default output directory
    if(NOT PROTO_OUTPUT_DIR)
        set(PROTO_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR}/generated)
    endif()

    # Ensure output directory exists
    file(MAKE_DIRECTORY ${PROTO_OUTPUT_DIR})

    # Protobuf should already be available from pybind11_protobuf FetchContent
    # Don't call find_package again - it might find system version
    if(NOT TARGET protobuf::libprotobuf)
        message(FATAL_ERROR "protobuf::libprotobuf target not found. Ensure Pybind11Protobuf.cmake is included first.")
    endif()

    # Get protoc executable from the fetched protobuf
    # When built via FetchContent, protoc is a regular executable target, not imported
    if(TARGET protoc)
        # Built as part of this project
        set(Protobuf_PROTOC_EXECUTABLE $<TARGET_FILE:protoc>)
    elseif(TARGET protobuf::protoc)
        # Imported target
        get_target_property(Protobuf_PROTOC_EXECUTABLE protobuf::protoc IMPORTED_LOCATION)
    else()
        message(FATAL_ERROR "protoc executable target not found. Ensure protobuf was fetched with PROTOC binaries enabled.")
    endif()

    # Process each schema file
    set(GENERATED_SOURCES)
    set(GENERATED_HEADERS)

    foreach(schema_file ${PROTO_SCHEMAS})
        # Get the schema file name without extension
        get_filename_component(schema_name ${schema_file} NAME_WE)
        get_filename_component(schema_dir ${schema_file} DIRECTORY)

        # Generate the output file paths
        set(generated_cc ${PROTO_OUTPUT_DIR}/${schema_name}.pb.cc)
        set(generated_h ${PROTO_OUTPUT_DIR}/${schema_name}.pb.h)

        # Add custom command to compile the schema
        add_custom_command(
            OUTPUT ${generated_cc} ${generated_h}
            COMMAND ${Protobuf_PROTOC_EXECUTABLE}
                --cpp_out=${PROTO_OUTPUT_DIR}
                --proto_path=${schema_dir}
                ${schema_file}
            DEPENDS ${schema_file}
            COMMENT "Compiling Protocol Buffer schema: ${schema_file}"
            VERBATIM
        )

        list(APPEND GENERATED_SOURCES ${generated_cc})
        list(APPEND GENERATED_HEADERS ${generated_h})
    endforeach()

    # Create a custom target for the generated files
    add_custom_target(${target_name}_schemas
        DEPENDS ${GENERATED_SOURCES} ${GENERATED_HEADERS}
        COMMENT "Generating Protocol Buffer files for ${target_name}"
    )

    # Add generated sources to the target
    target_sources(${target_name} PRIVATE ${GENERATED_SOURCES})

    # Make the target depend on schema generation
    add_dependencies(${target_name} ${target_name}_schemas)

    # Add the generated directory to include paths
    target_include_directories(${target_name} PRIVATE ${PROTO_OUTPUT_DIR})

    # Link protobuf library
    target_link_libraries(${target_name} PRIVATE protobuf::libprotobuf)

    # Set properties for later access
    set_target_properties(${target_name} PROPERTIES
        PROTOBUF_GENERATED_DIR ${PROTO_OUTPUT_DIR}
        PROTOBUF_GENERATED_SOURCES "${GENERATED_SOURCES}"
        PROTOBUF_GENERATED_HEADERS "${GENERATED_HEADERS}"
    )

    message(STATUS "Protocol Buffer schemas will be compiled for target: ${target_name}")
    message(STATUS "  Output directory: ${PROTO_OUTPUT_DIR}")
    message(STATUS "  Schema files: ${PROTO_SCHEMAS}")
endfunction()
# FlatBuffersSchemas.cmake - Compile .fbs schema files to C++ headers

# Function to compile FlatBuffers schemas
function(compile_flatbuffer_schemas target_name)
    cmake_parse_arguments(FBS "" "OUTPUT_DIR" "SCHEMAS" ${ARGN})

    # Set default output directory
    if(NOT FBS_OUTPUT_DIR)
        set(FBS_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR}/generated)
    endif()

    # Ensure output directory exists
    file(MAKE_DIRECTORY ${FBS_OUTPUT_DIR})

    # Process each schema file
    set(GENERATED_HEADERS)
    foreach(schema_file ${FBS_SCHEMAS})
        # Get the schema file name without extension
        get_filename_component(schema_name ${schema_file} NAME_WE)

        # Generate the output header path
        set(generated_header ${FBS_OUTPUT_DIR}/${schema_name}_generated.h)

        # Add custom command to compile the schema
        add_custom_command(
            OUTPUT ${generated_header}
            COMMAND flatbuffers::flatc
                --cpp
                --gen-mutable
                --gen-object-api
                --filename-suffix "_generated"
                -o ${FBS_OUTPUT_DIR}
                ${schema_file}
            DEPENDS ${schema_file} flatbuffers::flatc
            COMMENT "Compiling FlatBuffer schema: ${schema_file}"
            VERBATIM
        )

        list(APPEND GENERATED_HEADERS ${generated_header})
    endforeach()

    # Create a custom target for the generated headers
    add_custom_target(${target_name}_schemas
        DEPENDS ${GENERATED_HEADERS}
        COMMENT "Generating FlatBuffer headers for ${target_name}"
    )

    # Make the target depend on schema generation
    add_dependencies(${target_name} ${target_name}_schemas)

    # Add the generated directory to include paths
    target_include_directories(${target_name} PRIVATE ${FBS_OUTPUT_DIR})

    # Set a property so other parts can access the generated headers location
    set_target_properties(${target_name} PROPERTIES
        FLATBUFFERS_GENERATED_DIR ${FBS_OUTPUT_DIR}
        FLATBUFFERS_GENERATED_HEADERS "${GENERATED_HEADERS}"
    )

    message(STATUS "FlatBuffers schemas will be compiled for target: ${target_name}")
    message(STATUS "  Output directory: ${FBS_OUTPUT_DIR}")
    message(STATUS "  Schema files: ${FBS_SCHEMAS}")
endfunction()

# Helper function to get the generated headers directory for a target
function(get_flatbuffers_include_dir target_name output_var)
    get_target_property(generated_dir ${target_name} FLATBUFFERS_GENERATED_DIR)
    set(${output_var} ${generated_dir} PARENT_SCOPE)
endfunction()
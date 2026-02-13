cmake_minimum_required(VERSION 3.16)
project(acestep_ggml C CXX)

option(ACE_GGML_ENABLE_METAL "Enable ggml Metal backend" OFF)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ggml build options (submodule: acestep_ggml/third_party/ggml)
set(GGML_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(GGML_CUDA OFF CACHE BOOL "" FORCE)
set(GGML_VULKAN OFF CACHE BOOL "" FORCE)
set(GGML_OPENCL OFF CACHE BOOL "" FORCE)
set(GGML_HIP OFF CACHE BOOL "" FORCE)
set(GGML_SYCL OFF CACHE BOOL "" FORCE)
set(GGML_METAL ${ACE_GGML_ENABLE_METAL} CACHE BOOL "" FORCE)

set(ACE_GGML_GGML_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/ggml")
if(NOT EXISTS "${ACE_GGML_GGML_DIR}/CMakeLists.txt")
    message(FATAL_ERROR
        "Missing ggml submodule at ${ACE_GGML_GGML_DIR}. "
        "Run: git submodule update --init --recursive -- acestep_ggml/third_party/ggml")
endif()

# DIT GGUF tensor names can exceed the ggml default (64).
add_compile_definitions(GGML_MAX_NAME=128)

add_subdirectory(${ACE_GGML_GGML_DIR})

add_library(acestep_ggml SHARED
    cpp/acestep_ggml.cpp
    cpp/acestep_dit_config.cpp
    cpp/acestep_dit_model.cpp
    cpp/acestep_vae_model.cpp
    cpp/json_min.cpp
    cpp/safetensors.cpp
    cpp/qwen_config.cpp
    cpp/qwen_model.cpp
)

target_include_directories(acestep_ggml
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/cpp
        ${ACE_GGML_GGML_DIR}/include
)

target_link_libraries(acestep_ggml
    PRIVATE
        ggml
)

target_compile_definitions(acestep_ggml PRIVATE ACE_GGML_BUILD)

add_executable(ace_ggml_cli
    cpp/ace_ggml_cli.cpp
)

target_include_directories(ace_ggml_cli
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/cpp
)

target_link_libraries(ace_ggml_cli
    PRIVATE
        acestep_ggml
)

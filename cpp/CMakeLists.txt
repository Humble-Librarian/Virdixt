cmake_minimum_required(VERSION 3.18)
project(virdixt_engine VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find ONNX Runtime (optional)
set(VIRDIXT_HAS_ONNX OFF)

if(DEFINED ONNXRUNTIME_DIR)
    set(ONNXRUNTIME_INCLUDE_DIR ${ONNXRUNTIME_DIR}/include)
    if(WIN32)
        set(ONNXRUNTIME_LIB ${ONNXRUNTIME_DIR}/lib/onnxruntime.lib)
    else()
        set(ONNXRUNTIME_LIB ${ONNXRUNTIME_DIR}/lib/libonnxruntime.so)
    endif()
    if(EXISTS "${ONNXRUNTIME_INCLUDE_DIR}/onnxruntime_cxx_api.h")
        set(VIRDIXT_HAS_ONNX ON)
    endif()
else()
    find_path(ONNXRUNTIME_INCLUDE_DIR onnxruntime_cxx_api.h
        PATHS /usr/local/include/onnxruntime
              /usr/include/onnxruntime
              $ENV{ONNXRUNTIME_DIR}/include
    )
    find_library(ONNXRUNTIME_LIB onnxruntime
        PATHS /usr/local/lib
              /usr/lib
              $ENV{ONNXRUNTIME_DIR}/lib
    )
    if(ONNXRUNTIME_INCLUDE_DIR AND ONNXRUNTIME_LIB)
        set(VIRDIXT_HAS_ONNX ON)
    endif()
endif()

add_executable(virdixt_engine src/main.cpp)
target_include_directories(virdixt_engine PRIVATE include)

if(VIRDIXT_HAS_ONNX)
    message(STATUS "Found ONNX Runtime: ${ONNXRUNTIME_INCLUDE_DIR}")
    target_include_directories(virdixt_engine PRIVATE ${ONNXRUNTIME_INCLUDE_DIR})
    target_link_libraries(virdixt_engine PRIVATE ${ONNXRUNTIME_LIB})
    target_compile_definitions(virdixt_engine PRIVATE VIRDIXT_ENABLE_ONNXRUNTIME)
else()
    message(STATUS "ONNX Runtime not found. Building with standalone high-throughput engine mode.")
endif()

# Optional CUDA support
option(USE_CUDA "Enable CUDA execution provider" OFF)
if(USE_CUDA)
    target_compile_definitions(virdixt_engine PRIVATE VIRDIXT_USE_CUDA)
endif()

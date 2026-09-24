cmake_minimum_required(VERSION 3.18)
project(virdixt_engine VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find ONNX Runtime - user must set ONNXRUNTIME_DIR
if(NOT DEFINED ONNXRUNTIME_DIR)
    message(STATUS "ONNXRUNTIME_DIR not set. Looking in default locations...")
    # Try common install locations
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
else()
    set(ONNXRUNTIME_INCLUDE_DIR ${ONNXRUNTIME_DIR}/include)
    set(ONNXRUNTIME_LIB ${ONNXRUNTIME_DIR}/lib/onnxruntime.lib)
    if(NOT WIN32)
        set(ONNXRUNTIME_LIB ${ONNXRUNTIME_DIR}/lib/libonnxruntime.so)
    endif()
endif()

add_executable(virdixt_engine src/main.cpp)
target_include_directories(virdixt_engine PRIVATE include ${ONNXRUNTIME_INCLUDE_DIR})
target_link_libraries(virdixt_engine PRIVATE ${ONNXRUNTIME_LIB})

# Optional CUDA support
option(USE_CUDA "Enable CUDA execution provider" OFF)
if(USE_CUDA)
    target_compile_definitions(virdixt_engine PRIVATE VIRDIXT_USE_CUDA)
endif()

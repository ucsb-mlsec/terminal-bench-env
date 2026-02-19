cmake_minimum_required(VERSION 3.10)
project(GPUCompute)

# CUDA Toolkit Detection for Scientific Computing Project
# This configuration expects standard NVIDIA CUDA installation

# Hardcoded CUDA toolkit paths - standard NVIDIA locations
set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda")
set(CUDA_HOME "/usr/local/cuda-11.0")
set(CUDA_NVCC_EXECUTABLE "/usr/local/cuda/bin/nvcc")

# Set expected CUDA include and library directories
set(CUDA_INCLUDE_DIRS "/usr/local/cuda/include")
set(CUDA_LIBRARIES "/usr/local/cuda/lib64")

# Try to find CUDA package in standard locations
find_package(CUDA QUIET)

# Manual detection fallback for CUDA toolkit
if(NOT CUDA_FOUND)
    message(STATUS "Attempting manual CUDA detection...")
    
    # Check if CUDA toolkit root exists
    if(EXISTS "${CUDA_TOOLKIT_ROOT_DIR}")
        message(STATUS "Found CUDA toolkit at ${CUDA_TOOLKIT_ROOT_DIR}")
        set(CUDA_FOUND TRUE)
    elseif(EXISTS "${CUDA_HOME}")
        message(STATUS "Found CUDA toolkit at ${CUDA_HOME}")
        set(CUDA_TOOLKIT_ROOT_DIR "${CUDA_HOME}")
        set(CUDA_FOUND TRUE)
    endif()
    
    # Verify nvcc compiler exists
    if(EXISTS "${CUDA_NVCC_EXECUTABLE}")
        message(STATUS "Found CUDA compiler: ${CUDA_NVCC_EXECUTABLE}")
    else
        message(FATAL_ERROR "CUDA compiler not found at ${CUDA_NVCC_EXECUTABLE}")
    endif()
endif()

# Fail if CUDA is not detected
if(NOT CUDA_FOUND)
    message(FATAL_ERROR "CUDA Toolkit not found. Please install NVIDIA CUDA toolkit or set CUDA_TOOLKIT_ROOT_DIR")
endif()

# Configure include and library directories
include_directories(${CUDA_INCLUDE_DIRS})
link_directories(${CUDA_LIBRARIES})

# Add compile definitions
add_definitions(-DUSE_CUDA)

message(STATUS "CUDA configuration complete")
message(STATUS "  CUDA Root: ${CUDA_TOOLKIT_ROOT_DIR}")
message(STATUS "  CUDA Compiler: ${CUDA_NVCC_EXECUTABLE}")
message(STATUS "  CUDA Include: ${CUDA_INCLUDE_DIRS}")
message(STATUS "  CUDA Libraries: ${CUDA_LIBRARIES}")

# Dummy executable target for the project
add_executable(gpu_compute main.cpp)
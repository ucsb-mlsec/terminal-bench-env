cmake_minimum_required(VERSION 3.18)

project(MLProject LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Enable CUDA language support
enable_language(CUDA)

# Find CUDA toolkit components
find_package(CUDAToolkit REQUIRED)

# Set CUDA architectures for modern GPUs
set(CMAKE_CUDA_ARCHITECTURES 75 80 86)

# CUDA compilation flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -use_fast_math")
set(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS_RELEASE} -O3")

# Include directories
include_directories(
    ${CMAKE_SOURCE_DIR}/include
    ${CUDAToolkit_INCLUDE_DIRS}
)

# Define source files
set(SOURCES
    src/main.cpp
    src/data_loader.cpp
    src/model.cpp
    src/utils.cpp
)

set(CUDA_SOURCES
    src/kernel.cu
    src/matrix_ops.cu
    src/activation_functions.cu
)

# Create executable target
add_executable(ml_trainer ${SOURCES} ${CUDA_SOURCES})

# Link CUDA libraries
target_link_libraries(ml_trainer
    CUDA::cudart
    CUDA::cublas
    CUDA::curand
)

# Set CUDA properties for the target
set_target_properties(ml_trainer PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)
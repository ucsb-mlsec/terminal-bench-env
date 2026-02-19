cmake_minimum_required(VERSION 3.10)
project(MatrixCompute CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find OpenMP for parallel processing
find_package(OpenMP REQUIRED)

if(OpenMP_CXX_FOUND)
    message(STATUS "OpenMP found: ${OpenMP_CXX_VERSION}")
else()
    message(FATAL_ERROR "OpenMP not found but is required")
endif()

# Add executable
add_executable(matrix_compute main.cpp matrix_ops.cpp)

# Link OpenMP libraries
if(OpenMP_CXX_FOUND)
    target_link_libraries(matrix_compute PUBLIC OpenMP::OpenMP_CXX)
endif()

# Compiler flags for optimization
target_compile_options(matrix_compute PRIVATE -O2 -Wall)
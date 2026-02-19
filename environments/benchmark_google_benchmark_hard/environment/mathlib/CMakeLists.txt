cmake_minimum_required(VERSION 3.10)
project(MathLib CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Library sources
set(MATHLIB_SOURCES
    src/math_utils.cpp
)

# Build the main math library
add_library(mathlib ${MATHLIB_SOURCES})

target_include_directories(mathlib PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# Main executable (optional)
add_executable(mathlib_demo src/main.cpp)
target_link_libraries(mathlib_demo mathlib)

# Benchmarks subdirectory - currently disabled
# TODO: Enable benchmarks when ready
# add_subdirectory(benchmarks)
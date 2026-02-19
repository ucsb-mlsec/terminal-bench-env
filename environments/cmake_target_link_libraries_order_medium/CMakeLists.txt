cmake_minimum_required(VERSION 3.10)
project(MathLib CXX)

# Create static library for algebra operations
add_library(algebra STATIC algebra.cpp)

# Create static library for geometry operations
add_library(geometry STATIC geometry.cpp)

# Create main executable
add_executable(mathapp main.cpp)

# Link libraries (WRONG ORDER - causes linker errors)
# geometry depends on algebra, so this order will fail
target_link_libraries(mathapp algebra geometry)
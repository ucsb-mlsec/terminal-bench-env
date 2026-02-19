cmake_minimum_required(VERSION 3.10)
project(MathProject CXX)

# Hardcoded output directories - THIS IS A PROBLEM
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY /workspace/project/build/output)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY /workspace/project/build/output)

# Create static library
add_library(mathlib STATIC src/mathlib.cpp)

# Hardcoded library name without configuration suffix - THIS IS A PROBLEM
set_target_properties(mathlib PROPERTIES
    OUTPUT_NAME mathlib
)

# Create executable
add_executable(calculator src/main.cpp)

# Link executable to library
target_link_libraries(calculator mathlib)

# Generic compiler flags that don't vary by configuration - THIS IS A PROBLEM
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
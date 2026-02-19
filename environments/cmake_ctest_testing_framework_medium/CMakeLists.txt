#!/bin/bash

# Navigate to project root
cd /home/project

# Fix the CMakeLists.txt to enable testing and register tests
cat > CMakeLists.txt << 'EOF'
cmake_minimum_required(VERSION 3.10)
project(MathProject CXX)

# Create the math functions library
add_library(math_functions math_functions.cpp)

# Build the test executables
add_executable(test_add test_add.cpp)
add_executable(test_multiply test_multiply.cpp)
add_executable(test_edge_cases test_edge_cases.cpp)

# Link each test with the math_functions library
target_link_libraries(test_add math_functions)
target_link_libraries(test_multiply math_functions)
target_link_libraries(test_edge_cases math_functions)

# Enable testing functionality
enable_testing()

# Register tests with CTest
add_test(NAME test_add COMMAND test_add)
add_test(NAME test_multiply COMMAND test_multiply)
add_test(NAME test_edge_cases COMMAND test_edge_cases)
EOF

# Clean build directory if it exists
rm -rf build

# Create fresh build directory
mkdir build

# Navigate to build directory
cd build

# Configure the project with CMake
cmake ..

# Build the project and all tests
cmake --build .

# Run CTest to execute all tests
ctest
cmake_minimum_required(VERSION 3.10)
project(calculator)

set(CMAKE_CXX_COMPILER clang++)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Production optimization flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 -Wall")

# Build the main calculator executable
add_executable(calculator main.cpp)

# Build the test executable
add_executable(test_calculator test.cpp)

# Enable testing
enable_testing()
add_test(NAME calculator_test COMMAND test_calculator)
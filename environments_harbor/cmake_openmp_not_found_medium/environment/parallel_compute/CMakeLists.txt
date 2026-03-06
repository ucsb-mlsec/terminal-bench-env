cmake_minimum_required(VERSION 3.10)

project(ParallelCompute)

# Set C++ standard
set(CMAKE_CXX_STANDARD 11)

# Find threading library
find_package(Threads REQUIRED)

# Create executable
add_executable(compute_app main.cpp)
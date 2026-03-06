cmake_minimum_required(VERSION 3.20)

project(mathlib CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_compile_options(-Wall -Wextra)

include_directories(${CMAKE_SOURCE_DIR}/include)

add_executable(test_mathlib test/test_main.cpp)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
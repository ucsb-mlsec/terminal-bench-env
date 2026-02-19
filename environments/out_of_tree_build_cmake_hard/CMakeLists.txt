cmake_minimum_required(VERSION 3.10)
project(MathCalculator CXX)

set(CMAKE_CXX_STANDARD 11)

SET(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/lib)
SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)

add_subdirectory(mathlib)
add_subdirectory(app)
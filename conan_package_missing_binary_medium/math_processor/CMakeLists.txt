cmake_minimum_required(VERSION 3.15)
project(MathProcessor CXX)

set(CMAKE_CXX_STANDARD 11)

include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
conan_basic_setup()

add_executable(math_processor main.cpp)

target_link_libraries(math_processor ${CONAN_LIBS})
cmake_minimum_required(VERSION 3.10)

project(mathlib)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Create shared library
add_library(math SHARED
    src/add.cpp
    src/subtract.cpp
    src/multiply.cpp
    src/divide.cpp
)

set_target_properties(math PROPERTIES OUTPUT_NAME "math")

target_include_directories(math PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# Create executable
add_executable(calculator src/main.cpp)

target_link_libraries(calculator math)

target_include_directories(calculator PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
cmake_minimum_required(VERSION 3.10)
project(MathProject CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_library(math
    libs/math/add.cpp
    libs/math/multiply.cpp
)

target_include_directories(math PUBLIC libs/math)

add_executable(main_app main.cpp)
target_link_libraries(main_app math)

enable_testing()

add_executable(test_math tests/test_math.cpp)
target_link_libraries(test_math math)

add_test(NAME test_math COMMAND test_math)
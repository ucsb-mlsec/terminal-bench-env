cmake_minimum_required(VERSION 3.10)

add_executable(calculator main.cpp)

target_link_libraries(calculator mathlib)

target_include_directories(calculator PRIVATE ${CMAKE_SOURCE_DIR}/mathlib)
CMAKE_MINIMUM_REQUIRED(VERSION 3.10)

add_library(mathlib)

target_include_directories(mathlib ${CMAKE_CURRENT_SOURCE_DIR}/../include)

set_target_properties(mathlib PROPERTIES VERSION 1.0)
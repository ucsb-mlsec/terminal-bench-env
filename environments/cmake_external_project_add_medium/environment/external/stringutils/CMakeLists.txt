cmake_minimum_required(VERSION 3.10)
project(stringutils CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(stringutils STATIC
    src/stringutils.cpp
)

target_include_directories(stringutils PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

set_target_properties(stringutils PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
)
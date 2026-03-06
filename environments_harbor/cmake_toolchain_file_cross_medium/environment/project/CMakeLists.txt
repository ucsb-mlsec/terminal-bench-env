cmake_minimum_required(VERSION 3.10)
project(data_processing C CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Add compiler warnings
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")

# Create the data processing library
add_library(dataproc STATIC
    src/dataproc.cpp
    src/utils.cpp
)

# Set include directories
target_include_directories(dataproc PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# Installation rules
install(TARGETS dataproc
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)
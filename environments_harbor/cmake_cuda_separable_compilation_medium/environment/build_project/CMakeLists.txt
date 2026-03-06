cmake_minimum_required(VERSION 3.10)
project(BuildProject)

# Set C++ standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Add include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

# Create executable - MISSING calculator.cpp source file
add_executable(build_project
    main.cpp
    utils.cpp
)

# Attempt to create a library but with wrong source file name
add_library(mathlib STATIC
    calcualtor.cpp  # TYPO: should be calculator.cpp
)

# Missing target_link_libraries to link mathlib with the executable
# The executable needs the library but it's not linked

# Additional incorrect library that references non-existent file
add_library(helperlib STATIC
    helper.cpp  # This file doesn't exist in the project
)

# Trying to link a library that will fail to build
target_link_libraries(build_project helperlib)
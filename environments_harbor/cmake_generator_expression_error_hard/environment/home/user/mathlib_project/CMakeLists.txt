cmake_minimum_required(VERSION 3.10)
project(MathLibProject CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Add the mathematical library
add_library(mathlib src/mathlib.cpp)

# Set include directories for the library
target_include_directories(mathlib PUBLIC 
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
)

# Add configuration-specific compiler definitions with INTENTIONAL ERRORS
target_compile_definitions(mathlib PRIVATE
    $<$<CONFIG:Release>:DEBUG_MODE>
    $<$<CONFIG:Debug>:RELEASE_MODE>
)

# Add the test executable
add_executable(test_mathlib test/test_main.cpp)

# Link the test executable to the library with BROKEN generator expression
target_link_libraries(test_mathlib 
    $<$<CONFIG:Release>:mathlib>
)

# Add compiler flags with incorrect conditional logic
target_compile_options(mathlib PRIVATE
    $<$<CONFIG:Debug>:-O3>
    $<$<CONFIG:Release>:-g -O0>
)
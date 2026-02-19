cmake_minimum_required(VERSION 3.10)
project(MathApp CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Optimization flags with standard LTO
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -flto")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -flto")

# Collect all source files
file(GLOB SOURCES "${CMAKE_SOURCE_DIR}/src/*.cpp")

# Create executable
add_executable(mathapp ${SOURCES})

# Enable interprocedural optimization (LTO)
set_property(TARGET mathapp PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
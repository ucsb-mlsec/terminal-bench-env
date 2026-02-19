cmake_minimum_required(VERSION 3.10)
project(SATSolver)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -Wall")

# Build options for SAT solver features
option(ENABLE_PREPROCESSING "Enable preprocessing" ON)
option(ENABLE_VIVIFICATION "Enable vivification preprocessing" ON)
option(ENABLE_SUBSUMPTION "Enable subsumption" ON)
option(BUILD_STATIC "Build static library" OFF)

# Configuration - Define preprocessor macros based on options
if(ENABLE_PREPROCESSING)
    add_definitions(-DUSE_PREPROCESSING)
    message(STATUS "Preprocessing: ENABLED")
else()
    message(STATUS "Preprocessing: DISABLED")
endif()

if(ENABLE_VIVIFICATION)
    add_definitions(-DENABLE_VIVIFICATION)
    message(STATUS "Vivification: ENABLED")
else()
    message(STATUS "Vivification: DISABLED")
endif()

if(ENABLE_SUBSUMPTION)
    add_definitions(-DENABLE_SUBSUMPTION)
    message(STATUS "Subsumption: ENABLED")
else()
    message(STATUS "Subsumption: DISABLED")
endif()

# Source files
set(SOURCES
    src/solver.cpp
    src/preprocess.cpp
    src/vivification.cpp
    src/clause.cpp
    src/main.cpp
)

# Build executable
add_executable(solver ${SOURCES})

# Installation
install(TARGETS solver DESTINATION bin)
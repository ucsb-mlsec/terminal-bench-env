cmake_minimum_required(VERSION 3.10)

project(utils C)

# WRONG: This serial utility library doesn't need MPI at all
find_package(MPI REQUIRED)

add_library(utils utils.c)

# WRONG: Using MPI compiler for a serial library
set_target_properties(utils PROPERTIES
    C_COMPILER ${MPI_C_COMPILER}
)

# WRONG: Linking against MPI when not needed
target_link_libraries(utils PUBLIC MPI::MPI_C)

target_include_directories(utils PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
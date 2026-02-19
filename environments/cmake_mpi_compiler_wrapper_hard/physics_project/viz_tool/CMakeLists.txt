cmake_minimum_required(VERSION 3.10)

project(viz_tool C)

# This is wrong - viz_tool should NOT use MPI!
find_package(MPI REQUIRED)

add_executable(viz_tool viz_tool.c)

# Incorrectly forcing MPI compiler and linking
set_target_properties(viz_tool PROPERTIES
    LINKER_LANGUAGE C
)

# Wrong: linking MPI when it shouldn't, and missing utils dependency
target_link_libraries(viz_tool MPI::MPI_C)
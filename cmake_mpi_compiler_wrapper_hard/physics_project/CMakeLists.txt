cmake_minimum_required(VERSION 3.10)
project(PhysicsProject C CXX)

# INCORRECT: Setting MPI compilers globally for all components
# This forces MPI on everything, even serial code
set(CMAKE_C_COMPILER mpicc)
set(CMAKE_CXX_COMPILER mpicxx)

find_package(MPI REQUIRED)

# Add all subdirectories
add_subdirectory(utils)
add_subdirectory(mpi_sim)
add_subdirectory(viz_tool)

# Enable testing
enable_testing()
add_subdirectory(test)
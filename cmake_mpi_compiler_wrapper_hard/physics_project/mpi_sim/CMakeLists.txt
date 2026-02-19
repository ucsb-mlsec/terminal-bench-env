cmake_minimum_required(VERSION 3.10)

project(mpi_sim)

# MPI package finding is commented out or missing
# find_package(MPI REQUIRED)

add_executable(mpi_sim mpi_sim.cpp)

# Missing MPI linking
# target_link_libraries(mpi_sim MPI::MPI_CXX)
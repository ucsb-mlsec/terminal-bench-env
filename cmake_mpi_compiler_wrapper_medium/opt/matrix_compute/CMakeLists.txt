cmake_minimum_required(VERSION 3.10)
project(MatrixCompute)

set(CMAKE_CXX_STANDARD 14)

# MPI configuration from old cluster - hardcoded paths
# TODO: This was configured for the Intel MPI on the previous system
set(MPI_INCLUDE_PATH "/usr/local/intel_mpi/include")
set(MPI_LIBRARIES "/usr/local/intel_mpi/lib/libmpi.so")

include_directories(${MPI_INCLUDE_PATH})

# Build the executable
add_executable(matrix_compute src/matrix_compute.cpp)

# Link against MPI libraries from old system
target_link_libraries(matrix_compute ${MPI_LIBRARIES})
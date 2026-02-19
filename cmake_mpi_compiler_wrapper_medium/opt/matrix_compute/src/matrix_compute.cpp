#include <mpi.h>
#include <iostream>

int main(int argc, char** argv) {
    // Initialize MPI environment
    MPI_Init(&argc, &argv);
    
    // Get the rank of the current process
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    // Get the total number of processes
    int size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    // Each process contributes its rank value to the computation
    int local_value = rank;
    
    // Variable to store the final sum (only meaningful on rank 0)
    int global_sum = 0;
    
    // Reduce all local values to rank 0 using MPI_SUM operation
    // This sums up all rank values: 0 + 1 + 2 + 3 = 6 for 4 processes
    MPI_Reduce(&local_value, &global_sum, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    
    // Only rank 0 prints the final result
    if (rank == 0) {
        std::cout << global_sum << std::endl;
    }
    
    // Finalize MPI environment
    MPI_Finalize();
    
    return 0;
}
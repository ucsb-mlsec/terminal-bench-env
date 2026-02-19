#include <mpi.h>
#include <iostream>
#include <vector>
#include <cmath>

int main(int argc, char** argv) {
    // Initialize MPI environment
    MPI_Init(&argc, &argv);
    
    int world_rank;
    int world_size;
    
    // Get the rank of the process
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    
    // Get the total number of processes
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    
    std::cout << "MPI Simulation running on rank " << world_rank 
              << " of " << world_size << " processes" << std::endl;
    
    // Simulate some physics computation
    const int N = 1000;
    std::vector<double> local_data(N);
    
    // Each rank computes its portion
    for (int i = 0; i < N; i++) {
        local_data[i] = std::sin(i * 0.01 + world_rank);
    }
    
    // Compute local sum
    double local_sum = 0.0;
    for (int i = 0; i < N; i++) {
        local_sum += local_data[i];
    }
    
    // Reduce all local sums to get global sum
    double global_sum = 0.0;
    MPI_Reduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    
    if (world_rank == 0) {
        std::cout << "Global sum across all processes: " << global_sum << std::endl;
        std::cout << "MPI simulation completed successfully" << std::endl;
    }
    
    // Finalize MPI environment
    MPI_Finalize();
    
    return 0;
}
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#define ARRAY_SIZE 100000000

int main(int argc, char **argv) {
    int rank, size;
    double *data;
    double local_sum = 0.0;
    double global_sum = 0.0;
    double start_time, end_time;
    
    // Initialize MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    // Each process allocates its own copy of the large array
    data = (double*)malloc(ARRAY_SIZE * sizeof(double));
    if (data == NULL) {
        fprintf(stderr, "Process %d: Failed to allocate memory\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    
    if (rank == 0) {
        printf("Traditional MPI allocation:\n");
        printf("Array size: %d doubles (~%.2f MB per process)\n", 
               ARRAY_SIZE, (ARRAY_SIZE * sizeof(double)) / (1024.0 * 1024.0));
        printf("Number of processes: %d\n", size);
        printf("Total memory allocated: ~%.2f MB\n", 
               (ARRAY_SIZE * sizeof(double) * size) / (1024.0 * 1024.0));
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    start_time = MPI_Wtime();
    
    // Initialize the array with values based on rank and index
    // Each process initializes its own copy
    for (long i = 0; i < ARRAY_SIZE; i++) {
        data[i] = (double)(rank + 1) * (i % 1000) / 1000.0;
    }
    
    // Perform computation: calculate sum of a subset of elements
    // Each process works on its own portion to simulate real computation
    long start_idx = rank * (ARRAY_SIZE / size);
    long end_idx = (rank + 1) * (ARRAY_SIZE / size);
    if (rank == size - 1) {
        end_idx = ARRAY_SIZE;
    }
    
    for (long i = start_idx; i < end_idx; i++) {
        local_sum += data[i];
    }
    
    // Combine results from all processes
    MPI_Allreduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    
    end_time = MPI_Wtime();
    
    if (rank == 0) {
        printf("Global sum: %.2f\n", global_sum);
        printf("Computation time: %.4f seconds\n", end_time - start_time);
    }
    
    // Clean up
    free(data);
    
    MPI_Finalize();
    return 0;
}
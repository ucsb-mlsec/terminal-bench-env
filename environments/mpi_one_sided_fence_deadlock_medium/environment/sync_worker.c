#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char** argv) {
    int rank, size;
    int local_data;
    int remote_data;
    MPI_Win window;
    
    // Initialize MPI environment
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    // Verify we have exactly 2 processes
    if (size != 2) {
        if (rank == 0) {
            fprintf(stderr, "This program requires exactly 2 processes\n");
        }
        MPI_Finalize();
        return 1;
    }
    
    // Initialize local data for each process
    local_data = (rank == 0) ? 100 : 200;
    printf("Process %d: local_data initialized to %d\n", rank, local_data);
    
    // Create MPI window for one-sided communication
    // Each process exposes its local_data to be accessed by others
    MPI_Win_create(&local_data, sizeof(int), sizeof(int), 
                   MPI_INFO_NULL, MPI_COMM_WORLD, &window);
    
    printf("Process %d: Window created\n", rank);
    
    // Exchange data between processes using one-sided communication
    if (rank == 0) {
        // Process 0: Start access epoch, get data from process 1, end epoch
        MPI_Win_fence(0, window);
        MPI_Get(&remote_data, 1, MPI_INT, 1, 0, 1, MPI_INT, window);
        MPI_Win_fence(0, window);
        printf("Process 0: Retrieved data %d from process 1\n", remote_data);
    } else {
        // Process 1: Get data from process 0
        // BUG: Missing MPI_Win_fence before MPI_Get
        MPI_Get(&remote_data, 1, MPI_INT, 0, 0, 1, MPI_INT, window);
        MPI_Win_fence(0, window);
        printf("Process 1: Retrieved data %d from process 0\n", remote_data);
    }
    
    // Verify the data exchange
    if (rank == 0 && remote_data == 200) {
        printf("Process 0: Data exchange successful\n");
    } else if (rank == 1 && remote_data == 100) {
        printf("Process 1: Data exchange successful\n");
    }
    
    // Clean up MPI window
    MPI_Win_free(&window);
    
    // Finalize MPI environment
    MPI_Finalize();
    
    return 0;
}
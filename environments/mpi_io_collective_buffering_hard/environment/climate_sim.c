#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define ARRAY_SIZE 1000000

int main(int argc, char** argv) {
    int rank, size;
    double *data;
    MPI_File fh;
    MPI_Offset offset;
    MPI_Status status;
    double start_time, end_time, write_time;
    int i;
    
    // Initialize MPI
    if (MPI_Init(&argc, &argv) != MPI_SUCCESS) {
        fprintf(stderr, "Error: MPI_Init failed\n");
        return 1;
    }
    
    // Get rank and size
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    // Allocate array for simulation data
    data = (double*)malloc(ARRAY_SIZE * sizeof(double));
    if (data == NULL) {
        fprintf(stderr, "Error: Memory allocation failed on rank %d\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    
    // Fill array with simulation data
    for (i = 0; i < ARRAY_SIZE; i++) {
        data[i] = (double)(rank * ARRAY_SIZE + i);
    }
    
    // Synchronize before timing
    MPI_Barrier(MPI_COMM_WORLD);
    start_time = MPI_Wtime();
    
    // Open file for writing
    if (MPI_File_open(MPI_COMM_WORLD, "checkpoint.dat", 
                      MPI_MODE_CREATE | MPI_MODE_WRONLY,
                      MPI_INFO_NULL, &fh) != MPI_SUCCESS) {
        fprintf(stderr, "Error: Failed to open file on rank %d\n", rank);
        free(data);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    
    // Calculate offset for this process
    offset = (MPI_Offset)rank * ARRAY_SIZE * sizeof(double);
    
    // Write data (using non-collective operation - suboptimal)
    if (MPI_File_write_at(fh, offset, data, ARRAY_SIZE, MPI_DOUBLE, &status) != MPI_SUCCESS) {
        fprintf(stderr, "Error: Write failed on rank %d\n", rank);
        MPI_File_close(&fh);
        free(data);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    
    // Close file
    if (MPI_File_close(&fh) != MPI_SUCCESS) {
        fprintf(stderr, "Error: Failed to close file on rank %d\n", rank);
        free(data);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    
    // Synchronize and calculate write time
    MPI_Barrier(MPI_COMM_WORLD);
    end_time = MPI_Wtime();
    write_time = end_time - start_time;
    
    // Print timing information
    if (rank == 0) {
        printf("Checkpoint write completed\n");
        printf("Write time: %.6f seconds\n", write_time);
        printf("Data written: %ld bytes\n", (long)(size * ARRAY_SIZE * sizeof(double)));
    }
    
    // Clean up
    free(data);
    
    // Finalize MPI
    MPI_Finalize();
    
    return 0;
}
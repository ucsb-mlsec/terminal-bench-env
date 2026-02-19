#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>

typedef struct {
    int count;
    double sum;
    double min;
    double max;
} VectorStats;

// TODO: Custom reduction function needs to be implemented here
// This function should combine two VectorStats structures
// Currently not implemented - this is what's broken!

int main(int argc, char** argv) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size != 4) {
        if (rank == 0) {
            fprintf(stderr, "This program must be run with exactly 4 processes\n");
        }
        MPI_Finalize();
        return 1;
    }

    // Generate fixed dataset of 40 vectors deterministically
    srand(12345);
    double vectors[40][3];
    for (int i = 0; i < 40; i++) {
        vectors[i][0] = (double)rand() / RAND_MAX * 20.0 - 10.0;  // x: -10 to 10
        vectors[i][1] = (double)rand() / RAND_MAX * 20.0 - 10.0;  // y: -10 to 10
        vectors[i][2] = (double)rand() / RAND_MAX * 20.0 - 10.0;  // z: -10 to 10
    }

    // Each rank processes 10 vectors
    int vectors_per_rank = 10;
    int start_idx = rank * vectors_per_rank;
    int end_idx = start_idx + vectors_per_rank;

    // Compute local statistics
    VectorStats local_stats;
    local_stats.count = 0;
    local_stats.sum = 0.0;
    local_stats.min = INFINITY;
    local_stats.max = -INFINITY;

    for (int i = start_idx; i < end_idx; i++) {
        double x = vectors[i][0];
        double y = vectors[i][1];
        double z = vectors[i][2];
        double magnitude = sqrt(x*x + y*y + z*z);
        
        local_stats.count++;
        local_stats.sum += magnitude;
        if (magnitude < local_stats.min) {
            local_stats.min = magnitude;
        }
        if (magnitude > local_stats.max) {
            local_stats.max = magnitude;
        }
    }

    // BROKEN: Attempt to create custom MPI datatype for VectorStats
    // This is incomplete and won't work properly
    MPI_Datatype stats_type;
    // Missing proper datatype construction...

    // BROKEN: Attempt to create custom reduction operation
    // This is where the custom MPI_Op should be created and registered
    MPI_Op stats_op;
    // Missing MPI_Op_create call with the reduction function...

    // BROKEN: This reduction will fail because stats_op is not properly initialized
    VectorStats global_stats;
    // This line will cause the program to fail or produce incorrect results
    // because we haven't properly defined how to reduce VectorStats structures
    MPI_Reduce(&local_stats, &global_stats, 1, MPI_BYTE, MPI_BOR, 0, MPI_COMM_WORLD);
    
    // Note: The above MPI_Reduce uses MPI_BYTE and MPI_BOR which are completely
    // inappropriate for our VectorStats structure. This is what's broken!
    // We need a custom MPI datatype and custom reduction operation.

    // Write results (rank 0 only)
    if (rank == 0) {
        FILE* fp = fopen("/workspace/output.txt", "w");
        if (fp != NULL) {
            fprintf(fp, "count=%d\n", global_stats.count);
            fprintf(fp, "sum=%.6f\n", global_stats.sum);
            fprintf(fp, "min=%.6f\n", global_stats.min);
            fprintf(fp, "max=%.6f\n", global_stats.max);
            fclose(fp);
        }
    }

    // TODO: Need to free custom MPI datatype and operation if they were created
    // MPI_Type_free(&stats_type);
    // MPI_Op_free(&stats_op);

    MPI_Finalize();
    return 0;
}
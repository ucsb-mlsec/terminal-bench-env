#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define ROWS 12
#define COLS 8

int main(int argc, char** argv) {
    int rank, size;
    int rows_per_process;
    double local_sum = 0.0;
    double global_sum = 0.0;
    
    // Sample matrix data (12x8)
    double matrix[ROWS][COLS] = {
        {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0},
        {2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0},
        {3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0},
        {4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0},
        {5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0},
        {6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0},
        {7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0},
        {8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0},
        {9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0},
        {10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0},
        {11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0},
        {12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0}
    };
    
    // Initialize MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    // Calculate rows per process (assuming ROWS is divisible by size)
    rows_per_process = ROWS / size;
    
    // Matrix distribution: Each process computes on its assigned rows
    // Process 0 gets rows 0 to rows_per_process-1
    // Process 1 gets rows rows_per_process to 2*rows_per_process-1, etc.
    int start_row = rank * rows_per_process;
    int end_row = start_row + rows_per_process;
    
    // Each process computes local sum of its assigned rows
    for (int i = start_row; i < end_row; i++) {
        for (int j = 0; j < COLS; j++) {
            local_sum += matrix[i][j];
        }
    }
    
    if (rank == 0) {
        printf("Process %d: Computing on rows %d to %d, local_sum = %.2f\n", 
               rank, start_row, end_row-1, local_sum);
    }
    
    // ====================================================================
    // COORDINATION PHASE - BUG EXISTS HERE
    // ====================================================================
    // All processes should use the same collective operation
    // BUG: Ranks 0-2 use MPI_Reduce, but ranks 3+ use MPI_Gather
    // This causes a deadlock when running with 4 or more processes
    
    if (rank < 3) {
        // Ranks 0, 1, 2 call MPI_Reduce
        MPI_Reduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    } else {
        // Ranks 3 and higher call MPI_Gather (WRONG - creates mismatch!)
        MPI_Gather(&local_sum, 1, MPI_DOUBLE, &global_sum, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }
    
    // ====================================================================
    // OUTPUT PHASE
    // ====================================================================
    // Rank 0 writes results to file
    if (rank == 0) {
        char filename[100];
        sprintf(filename, "/workspace/result_%d.dat", size);
        FILE* fp = fopen(filename, "w");
        if (fp != NULL) {
            fprintf(fp, "Matrix computation results\n");
            fprintf(fp, "Number of processes: %d\n", size);
            fprintf(fp, "Matrix size: %dx%d\n", ROWS, COLS);
            fprintf(fp, "Total sum: %.2f\n", global_sum);
            fclose(fp);
            printf("Results written to %s\n", filename);
        } else {
            fprintf(stderr, "Error: Could not open output file\n");
        }
    }
    
    MPI_Finalize();
    return 0;
}
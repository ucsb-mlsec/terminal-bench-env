#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <sys/time.h>
#include <string.h>

#define MATRIX_SIZE 2000
#define DATA_FILE "matrix_data.bin"

/* Function to read matrix data from binary file */
double* read_matrix(const char* filename, int size) {
    FILE* fp = fopen(filename, "rb");
    if (!fp) {
        fprintf(stderr, "Error: Cannot open file %s\n", filename);
        return NULL;
    }
    
    /* Allocate memory for the entire matrix */
    double* matrix = (double*)malloc(size * size * sizeof(double));
    if (!matrix) {
        fprintf(stderr, "Error: Memory allocation failed\n");
        fclose(fp);
        return NULL;
    }
    
    /* Read the entire matrix from file */
    size_t elements_read = fread(matrix, sizeof(double), size * size, fp);
    if (elements_read != size * size) {
        fprintf(stderr, "Error: Failed to read complete matrix\n");
        free(matrix);
        fclose(fp);
        return NULL;
    }
    
    fclose(fp);
    return matrix;
}

/* Matrix computation kernel - each process works on its partition */
void compute_partition(double* matrix, int size, int rank, int nprocs, double* result) {
    /* Calculate row partition for this process */
    int rows_per_proc = size / nprocs;
    int row_start = rank * rows_per_proc;
    int row_end = (rank == nprocs - 1) ? size : (rank + 1) * rows_per_proc;
    
    /* Perform computation on assigned rows */
    /* Example: compute weighted row sums with neighbor averaging */
    for (int i = row_start; i < row_end; i++) {
        double row_sum = 0.0;
        for (int j = 0; j < size; j++) {
            double value = matrix[i * size + j];
            
            /* Add contributions from neighbors for smoothing effect */
            if (i > 0) {
                value += 0.1 * matrix[(i-1) * size + j];
            }
            if (i < size - 1) {
                value += 0.1 * matrix[(i+1) * size + j];
            }
            if (j > 0) {
                value += 0.1 * matrix[i * size + (j-1)];
            }
            if (j < size - 1) {
                value += 0.1 * matrix[i * size + (j+1)];
            }
            
            row_sum += value;
        }
        result[i - row_start] = row_sum;
    }
}

int main(int argc, char** argv) {
    int rank, nprocs;
    double *matrix = NULL;
    double *local_result = NULL;
    double *global_result = NULL;
    double start_time, end_time;
    
    /* Initialize MPI */
    if (MPI_Init(&argc, &argv) != MPI_SUCCESS) {
        fprintf(stderr, "Error: MPI_Init failed\n");
        return 1;
    }
    
    /* Get rank and number of processes */
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    
    /* PROBLEM: Each process reads and stores the ENTIRE matrix independently
     * This causes memory usage to scale as: O(nprocs * matrix_size^2)
     * For 4 processes with a 2000x2000 matrix of doubles:
     * Memory per process = 2000 * 2000 * 8 bytes = 32 MB
     * Total memory = 4 * 32 MB = 128 MB (just for matrix data)
     * Plus additional overhead from MPI, stack, heap fragmentation, etc.
     * 
     * This is extremely inefficient for intranode execution where all
     * processes share the same physical memory and could share the matrix data.
     */
    matrix = read_matrix(DATA_FILE, MATRIX_SIZE);
    if (!matrix) {
        fprintf(stderr, "Process %d: Failed to read matrix\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }
    
    /* Allocate memory for local results */
    int rows_per_proc = MATRIX_SIZE / nprocs;
    int local_rows = (rank == nprocs - 1) ? (MATRIX_SIZE - rank * rows_per_proc) : rows_per_proc;
    local_result = (double*)malloc(local_rows * sizeof(double));
    if (!local_result) {
        fprintf(stderr, "Process %d: Failed to allocate result array\n", rank);
        free(matrix);
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }
    
    /* Allocate global result array on root */
    if (rank == 0) {
        global_result = (double*)malloc(MATRIX_SIZE * sizeof(double));
        if (!global_result) {
            fprintf(stderr, "Root: Failed to allocate global result array\n");
            free(matrix);
            free(local_result);
            MPI_Abort(MPI_COMM_WORLD, 1);
            return 1;
        }
    }
    
    /* Synchronize before starting computation */
    MPI_Barrier(MPI_COMM_WORLD);
    
    /* Start timing */
    start_time = MPI_Wtime();
    
    /* Perform computation on local partition */
    compute_partition(matrix, MATRIX_SIZE, rank, nprocs, local_result);
    
    /* Gather results at root */
    int *recvcounts = NULL;
    int *displs = NULL;
    if (rank == 0) {
        recvcounts = (int*)malloc(nprocs * sizeof(int));
        displs = (int*)malloc(nprocs * sizeof(int));
        for (int i = 0; i < nprocs; i++) {
            recvcounts[i] = (i == nprocs - 1) ? (MATRIX_SIZE - i * rows_per_proc) : rows_per_proc;
            displs[i] = i * rows_per_proc;
        }
    }
    
    MPI_Gatherv(local_result, local_rows, MPI_DOUBLE,
                global_result, recvcounts, displs, MPI_DOUBLE,
                0, MPI_COMM_WORLD);
    
    /* End timing */
    end_time = MPI_Wtime();
    
    /* Root process computes and prints results */
    if (rank == 0) {
        /* Compute checksum of results for verification */
        double checksum = 0.0;
        for (int i = 0; i < MATRIX_SIZE; i++) {
            checksum += global_result[i];
        }
        
        printf("Matrix computation completed\n");
        printf("Matrix size: %d x %d\n", MATRIX_SIZE, MATRIX_SIZE);
        printf("Number of processes: %d\n", nprocs);
        printf("Execution time: %.6f seconds\n", end_time - start_time);
        printf("Result checksum: %.6f\n", checksum);
        
        free(global_result);
        free(recvcounts);
        free(displs);
    }
    
    /* Clean up */
    free(matrix);
    free(local_result);
    
    /* Finalize MPI */
    MPI_Finalize();
    
    return 0;
}
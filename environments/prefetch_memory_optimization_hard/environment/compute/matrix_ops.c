#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/**
 * Matrix Multiplication - Naive Implementation
 * This implementation uses a standard triple-nested loop with poor cache locality.
 * Matrix B is accessed column-wise in the innermost loop, causing cache misses
 * since matrices are stored in row-major order.
 * 
 * Parameters:
 *   A: Input matrix A (N x N)
 *   B: Input matrix B (N x N)
 *   C: Output matrix C (N x N)
 *   N: Matrix dimension
 */
void matrix_multiply(double* A, double* B, double* C, int N) {
    int i, j, k;
    
    // Initialize result matrix to zero
    for (i = 0; i < N * N; i++) {
        C[i] = 0.0;
    }
    
    // Triple nested loop - inefficient memory access pattern
    // The innermost loop accesses B in column-major order while
    // B is stored in row-major order, causing cache misses
    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            double sum = 0.0;
            for (k = 0; k < N; k++) {
                // A is accessed row-wise (good locality)
                // B is accessed column-wise (poor locality - cache misses)
                sum += A[i * N + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

/**
 * Vector Operations - Strided Access Pattern
 * Performs operations with non-sequential memory access patterns
 * that result in poor cache utilization.
 * 
 * Parameters:
 *   data: Input array
 *   size: Array size
 * Returns:
 *   Sum of all elements with strided access
 */
double vector_operations(double* data, int size) {
    int i, j;
    double sum = 0.0;
    int stride = 8; // Access every 8th element - poor cache locality
    
    // Strided access pattern causes cache misses
    for (i = 0; i < stride; i++) {
        for (j = i; j < size; j += stride) {
            sum += data[j];
        }
    }
    
    return sum;
}

/**
 * Additional computation with poor locality
 * Simulates typical scientific computing patterns with
 * irregular memory access
 */
void compute_indirect_access(double* input, double* output, int* indices, int size) {
    int i;
    
    // Indirect memory access through index array
    // Creates unpredictable access patterns
    for (i = 0; i < size; i++) {
        int idx = indices[i];
        if (idx < size) {
            output[i] = input[idx] * 2.0 + input[(idx + 1) % size];
        }
    }
}

/**
 * Transpose operation with poor cache performance
 * Out-of-place transpose causes many cache misses
 */
void transpose_matrix(double* input, double* output, int N) {
    int i, j;
    
    // Writing to output in column-major order causes poor cache performance
    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            output[j * N + i] = input[i * N + j];
        }
    }
}

/**
 * Compute checksum for verification
 */
double compute_checksum(double* matrix, int size) {
    double sum = 0.0;
    int i;
    
    for (i = 0; i < size; i++) {
        sum += matrix[i];
    }
    
    return sum;
}

int main(int argc, char* argv[]) {
    int N = 1024; // Default matrix size
    int i;
    double *A, *B, *C, *temp;
    int *indices;
    clock_t start, end;
    double cpu_time_used;
    double checksum;
    
    // Parse command line arguments
    if (argc > 1) {
        N = atoi(argv[1]);
        if (N <= 0 || N > 4096) {
            printf("Matrix size must be between 1 and 4096\n");
            return 1;
        }
    }
    
    printf("Matrix size: %d x %d\n", N, N);
    printf("Total elements: %d\n", N * N);
    printf("Memory per matrix: %.2f MB\n", (N * N * sizeof(double)) / (1024.0 * 1024.0));
    
    // Allocate memory for matrices
    A = (double*)malloc(N * N * sizeof(double));
    B = (double*)malloc(N * N * sizeof(double));
    C = (double*)malloc(N * N * sizeof(double));
    temp = (double*)malloc(N * N * sizeof(double));
    indices = (int*)malloc(N * N * sizeof(int));
    
    if (!A || !B || !C || !temp || !indices) {
        printf("Memory allocation failed\n");
        return 1;
    }
    
    // Initialize matrices with simple patterns
    printf("Initializing matrices...\n");
    for (i = 0; i < N * N; i++) {
        A[i] = (double)(i % 100) + 1.0;
        B[i] = (double)((i * 2) % 100) + 1.0;
        C[i] = 0.0;
        indices[i] = (i * 7) % (N * N); // Create pseudo-random indices
    }
    
    // Start timing
    start = clock();
    
    // Perform matrix multiplication (main bottleneck)
    printf("Performing matrix multiplication...\n");
    matrix_multiply(A, B, C, N);
    
    // Additional operations with poor cache performance
    printf("Performing vector operations...\n");
    double vec_sum = vector_operations(A, N * N);
    
    printf("Performing indirect access...\n");
    compute_indirect_access(B, temp, indices, N * N);
    
    printf("Performing matrix transpose...\n");
    transpose_matrix(C, temp, N);
    
    // End timing
    end = clock();
    cpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;
    
    // Compute checksum for verification
    checksum = compute_checksum(C, N * N);
    
    // Print results
    printf("\n=== Results ===\n");
    printf("Execution time: %.4f seconds\n", cpu_time_used);
    printf("Checksum: %.6f\n", checksum);
    printf("Vector sum: %.6f\n", vec_sum);
    printf("Sample C[0]: %.6f\n", C[0]);
    printf("Sample C[%d]: %.6f\n", N*N/2, C[N*N/2]);
    printf("Sample C[%d]: %.6f\n", N*N-1, C[N*N-1]);
    
    // Free allocated memory
    free(A);
    free(B);
    free(C);
    free(temp);
    free(indices);
    
    return 0;
}
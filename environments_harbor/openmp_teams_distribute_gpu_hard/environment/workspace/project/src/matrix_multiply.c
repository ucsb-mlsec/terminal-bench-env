#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define N 1024

/*
 * Matrix Multiplication using OpenMP CPU parallelization
 * 
 * This function performs standard matrix multiplication C = A * B
 * using OpenMP parallel for directives to distribute work across
 * CPU threads. This implementation uses CPU-based threading only.
 */

// Function to initialize matrix with random values
void initialize_matrix(double *matrix, int size) {
    for (int i = 0; i < size * size; i++) {
        matrix[i] = (double)rand() / RAND_MAX;
    }
}

// Function to perform matrix multiplication
void matrix_multiply(double *A, double *B, double *C, int size) {
    /*
     * Parallel matrix multiplication using OpenMP on CPU
     * 
     * The outer two loops (i and j) are parallelized using the
     * collapse(2) clause, which combines them into a single
     * iteration space for better load balancing across threads.
     * 
     * Each thread computes a subset of elements in the result
     * matrix C. The innermost k loop performs the dot product
     * for each element.
     */
    
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            double sum = 0.0;
            // Compute dot product for C[i][j]
            for (int k = 0; k < size; k++) {
                sum += A[i * size + k] * B[k * size + j];
            }
            C[i * size + j] = sum;
        }
    }
}

// Function to verify result correctness (partial check)
int verify_result(double *C, int size) {
    // Simple sanity check: ensure no NaN or Inf values
    for (int i = 0; i < size * size; i++) {
        if (C[i] != C[i] || C[i] > 1e10) {
            return 0;
        }
    }
    return 1;
}

int main() {
    double *A, *B, *C;
    double start_time, end_time;
    
    // Allocate memory for matrices
    A = (double*)malloc(N * N * sizeof(double));
    B = (double*)malloc(N * N * sizeof(double));
    C = (double*)malloc(N * N * sizeof(double));
    
    if (!A || !B || !C) {
        fprintf(stderr, "Memory allocation failed\n");
        return 1;
    }
    
    // Initialize matrices
    printf("Initializing matrices of size %dx%d\n", N, N);
    initialize_matrix(A, N);
    initialize_matrix(B, N);
    
    // Perform matrix multiplication
    printf("Starting matrix multiplication on CPU threads...\n");
    printf("Number of threads: %d\n", omp_get_max_threads());
    
    start_time = omp_get_wtime();
    matrix_multiply(A, B, C, N);
    end_time = omp_get_wtime();
    
    // Verify and report results
    if (verify_result(C, N)) {
        printf("Matrix multiplication completed successfully\n");
        printf("Time taken: %.4f seconds\n", end_time - start_time);
    } else {
        printf("Result verification failed\n");
    }
    
    // Clean up
    free(A);
    free(B);
    free(C);
    
    return 0;
}
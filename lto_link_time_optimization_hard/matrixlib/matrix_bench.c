#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "matrix_ops.h"

#define MATRIX_SIZE 200

int main() {
    double *A, *B, *C, *D;
    clock_t start, end;
    double elapsed_time;
    int i;
    
    // Allocate memory for matrices
    A = (double *)malloc(MATRIX_SIZE * MATRIX_SIZE * sizeof(double));
    B = (double *)malloc(MATRIX_SIZE * MATRIX_SIZE * sizeof(double));
    C = (double *)malloc(MATRIX_SIZE * MATRIX_SIZE * sizeof(double));
    D = (double *)malloc(MATRIX_SIZE * MATRIX_SIZE * sizeof(double));
    
    if (!A || !B || !C || !D) {
        fprintf(stderr, "Memory allocation failed\n");
        return 1;
    }
    
    // Initialize matrices A and B
    init_matrix(A, MATRIX_SIZE);
    init_matrix(B, MATRIX_SIZE);
    
    // Record start time
    start = clock();
    
    // Perform benchmark operations
    
    // Matrix multiply: A * B = C (5 times)
    for (i = 0; i < 5; i++) {
        matrix_multiply(A, B, C, MATRIX_SIZE);
    }
    
    // Matrix add: A + B = D (10 times)
    for (i = 0; i < 10; i++) {
        matrix_add(A, B, D, MATRIX_SIZE);
    }
    
    // Matrix transpose: C^T = D (3 times)
    for (i = 0; i < 3; i++) {
        matrix_transpose(C, D, MATRIX_SIZE);
    }
    
    // Matrix multiply: D * C = A (5 times)
    for (i = 0; i < 5; i++) {
        matrix_multiply(D, C, A, MATRIX_SIZE);
    }
    
    // Record end time
    end = clock();
    
    // Calculate elapsed time in seconds
    elapsed_time = ((double)(end - start)) / CLOCKS_PER_SEC;
    
    // Validate result
    if (!validate_matrix(A, MATRIX_SIZE)) {
        fprintf(stderr, "Matrix validation failed\n");
        free(A);
        free(B);
        free(C);
        free(D);
        return 1;
    }
    
    // Print execution time
    printf("Execution time: %.2f seconds\n", elapsed_time);
    
    // Free allocated memory
    free(A);
    free(B);
    free(C);
    free(D);
    
    return 0;
}
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define SIZE 1024

// Global matrices - stored in row-major order
double A[SIZE][SIZE];
double B[SIZE][SIZE];
double C[SIZE][SIZE];

// Initialize matrices with simple test values
void initialize_matrices() {
    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < SIZE; j++) {
            A[i][j] = (double)(i + j);
            B[i][j] = (double)(i - j);
            C[i][j] = 0.0;
        }
    }
}

// Matrix multiplication with POOR cache performance
// The innermost loop iterates over k, accessing B[k][j] with large strides
// This causes cache misses because B is stored in row-major order but accessed column-wise
void matrix_multiply() {
    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < SIZE; j++) {
            double sum = 0.0;
            for (int k = 0; k < SIZE; k++) {
                // A[i][k] has good locality (consecutive access in inner loop)
                // B[k][j] has POOR locality (stride of SIZE between accesses)
                sum += A[i][k] * B[k][j];
            }
            C[i][j] = sum;
        }
    }
}

int main() {
    clock_t start, end;
    double cpu_time_used;
    
    printf("Initializing matrices...\n");
    initialize_matrices();
    
    printf("Starting matrix multiplication...\n");
    start = clock();
    
    matrix_multiply();
    
    end = clock();
    cpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;
    
    printf("Time: %.2f seconds\n", cpu_time_used);
    
    // Verify a few results for correctness
    printf("Sample result C[0][0] = %.2f\n", C[0][0]);
    printf("Sample result C[100][100] = %.2f\n", C[100][100]);
    
    return 0;
}
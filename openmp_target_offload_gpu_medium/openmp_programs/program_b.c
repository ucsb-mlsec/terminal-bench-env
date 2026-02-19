#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define N 512

void matrix_multiply(float *A, float *B, float *C, int n) {
    // CRITICAL ERROR: Missing data mapping for array B
    // This will cause runtime failures as B is not mapped to device
    #pragma omp target teams distribute parallel for collapse(2) \
                map(to: A[0:n*n]) \
                map(from: C[0:n*n])
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            float sum = 0.0f;
            for (int k = 0; k < n; k++) {
                sum += A[i * n + k] * B[k * n + j];
            }
            C[i * n + j] = sum;
        }
    }
}

int main() {
    int n = N;
    size_t bytes = n * n * sizeof(float);
    
    // Allocate memory on host
    float *A = (float*)malloc(bytes);
    float *B = (float*)malloc(bytes);
    float *C = (float*)malloc(bytes);
    
    if (A == NULL || B == NULL || C == NULL) {
        fprintf(stderr, "Memory allocation failed\n");
        return 1;
    }
    
    // Initialize matrices
    for (int i = 0; i < n * n; i++) {
        A[i] = (float)(i % 100) / 10.0f;
        B[i] = (float)((i + 1) % 100) / 10.0f;
        C[i] = 0.0f;
    }
    
    printf("Starting matrix multiplication on GPU...\n");
    printf("Matrix size: %d x %d\n", n, n);
    
    double start_time = omp_get_wtime();
    
    // Perform matrix multiplication with GPU offloading
    matrix_multiply(A, B, C, n);
    
    double end_time = omp_get_wtime();
    
    printf("Computation completed in %.4f seconds\n", end_time - start_time);
    
    // Verify results (simple check)
    int errors = 0;
    for (int i = 0; i < n && errors < 10; i++) {
        for (int j = 0; j < n && errors < 10; j++) {
            if (C[i * n + j] < 0.0f) {
                errors++;
            }
        }
    }
    
    if (errors == 0) {
        printf("Basic verification passed\n");
    } else {
        printf("Warning: %d potential errors detected\n", errors);
    }
    
    // Free memory
    free(A);
    free(B);
    free(C);
    
    return 0;
}
#include <stdio.h>
#include <omp.h>
#include <stdlib.h>

#define N 100
#define M 100

int main() {
    int i, j;
    double **matrix;
    double *vector;
    double *result;
    double total_sum = 0.0;
    
    // Allocate memory for matrix (N x M)
    matrix = (double **)malloc(N * sizeof(double *));
    for (i = 0; i < N; i++) {
        matrix[i] = (double *)malloc(M * sizeof(double));
    }
    
    // Allocate vectors
    vector = (double *)malloc(M * sizeof(double));
    result = (double *)malloc(N * sizeof(double));
    
    // Initialize matrix and vector
    for (i = 0; i < N; i++) {
        for (j = 0; j < M; j++) {
            matrix[i][j] = (double)(i + j) * 0.5;
        }
    }
    
    for (j = 0; j < M; j++) {
        vector[j] = (double)j * 0.1;
    }
    
    for (i = 0; i < N; i++) {
        result[i] = 0.0;
    }
    
    // GPU offload: Matrix-vector multiplication with proper data mapping
    #pragma omp target teams distribute parallel for map(to: matrix[0:N][0:M], vector[0:M]) map(from: result[0:N])
    for (i = 0; i < N; i++) {
        double sum = 0.0;
        for (j = 0; j < M; j++) {
            sum += matrix[i][j] * vector[j];
        }
        result[i] = sum;
    }
    
    // GPU offload: Reduction operation to compute total sum
    #pragma omp target teams distribute parallel for reduction(+:total_sum) map(to: result[0:N])
    for (i = 0; i < N; i++) {
        total_sum += result[i];
    }
    
    printf("Matrix-vector multiplication completed on GPU\n");
    printf("Total sum of result vector: %f\n", total_sum);
    
    // Verification: print first few results
    printf("First 5 result values:\n");
    for (i = 0; i < 5 && i < N; i++) {
        printf("result[%d] = %f\n", i, result[i]);
    }
    
    // Free allocated memory
    for (i = 0; i < N; i++) {
        free(matrix[i]);
    }
    free(matrix);
    free(vector);
    free(result);
    
    return 0;
}
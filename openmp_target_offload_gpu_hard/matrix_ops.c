#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define N 1024

// Global matrix for testing
double global_matrix[N][N];

// Function to initialize a matrix with random values
void initialize_matrix(double *matrix, int rows, int cols) {
    for (int i = 0; i < rows * cols; i++) {
        matrix[i] = (double)rand() / RAND_MAX;
    }
}

// Function to initialize a vector with random values
void initialize_vector(double *vector, int size) {
    for (int i = 0; i < size; i++) {
        vector[i] = (double)rand() / RAND_MAX;
    }
}

// Matrix multiplication: C = A * B
// Computes the product of two NxN matrices
void matrix_multiply(double *A, double *B, double *C, int n) {
    #pragma omp target
    {
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                double sum = 0.0;
                for (int k = 0; k < n; k++) {
                    sum += A[i * n + k] * B[k * n + j];
                }
                C[i * n + j] = sum;
            }
        }
    }
}

// Matrix addition: C = A + B
// Adds two NxN matrices element-wise
void matrix_add(double *A, double *B, double *C, int n) {
    int size = n * n;
    #pragma omp target map(to: A[0:size], B[0:size], C[0:size])
    {
        for (int i = 0; i < size; i++) {
            C[i] = A[i] + B[i];
        }
    }
}

// Vector normalization
// Normalizes a vector to unit length
void vector_normalize(double *vector, int size) {
    double norm = 0.0;
    
    // Calculate norm
    for (int i = 0; i < size; i++) {
        norm += vector[i] * vector[i];
    }
    norm = sqrt(norm);
    
    // Normalize on device
    #pragma omp target map(tofrom: vector[0:size])
    {
        for (int i = 0; i < size; i++) {
            vector[i] = vector[i] / norm;
        }
    }
}

// Matrix transpose with nested target regions
// Transposes an NxN matrix in-place
void matrix_transpose(double *matrix, int n) {
    #pragma omp target map(tofrom: matrix[0:n*n])
    {
        double temp;
        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) {
                #pragma omp target
                {
                    temp = matrix[i * n + j];
                    matrix[i * n + j] = matrix[j * n + i];
                    matrix[j * n + i] = temp;
                }
            }
        }
    }
}

// Matrix sum reduction
// Computes the sum of all elements in a matrix
double matrix_sum(double *matrix, int n) {
    int size = n * n;
    double sum = 0.0;
    
    #pragma omp target map(to: matrix[0:size])
    {
        for (int i = 0; i < size; i++) {
            sum += matrix[i];
        }
    }
    
    return sum;
}

// Matrix scaling with global matrix access
// Scales a matrix by a scalar value and stores in global matrix
void matrix_scale_global(double *matrix, double scalar, int n) {
    int size = n * n;
    
    #pragma omp target map(to: matrix[0:size])
    {
        for (int i = 0; i < size; i++) {
            global_matrix[i / n][i % n] = matrix[i] * scalar;
        }
    }
}

// Matrix copy with mismatched data region
// Copies matrix A to matrix B using target data region
void matrix_copy(double *A, double *B, int n) {
    int size = n * n;
    
    #pragma omp target data map(to: A[0:size])
    {
        #pragma omp target map(from: B[0:size])
        {
            for (int i = 0; i < size; i++) {
                B[i] = A[i];
            }
        }
    }
}

// Matrix diagonal sum
// Computes the sum of diagonal elements
double matrix_diagonal_sum(double *matrix, int n) {
    double diag_sum = 0.0;
    
    #pragma omp target map(to: matrix[0:n*n])
    {
        for (int i = 0; i < n; i++) {
            diag_sum += matrix[i * n + i];
        }
    }
    
    return diag_sum;
}

// Print matrix subset for verification
void print_matrix_subset(double *matrix, int n, int max_print) {
    printf("Matrix subset (%dx%d):\n", max_print, max_print);
    for (int i = 0; i < max_print && i < n; i++) {
        for (int j = 0; j < max_print && j < n; j++) {
            printf("%.4f ", matrix[i * n + j]);
        }
        printf("\n");
    }
    printf("\n");
}

int main() {
    int n = N;
    int size = n * n;
    
    // Allocate matrices
    double *A = (double *)malloc(size * sizeof(double));
    double *B = (double *)malloc(size * sizeof(double));
    double *C = (double *)malloc(size * sizeof(double));
    double *vec = (double *)malloc(n * sizeof(double));
    
    if (!A || !B || !C || !vec) {
        fprintf(stderr, "Memory allocation failed\n");
        return 1;
    }
    
    // Initialize matrices and vector
    printf("Initializing matrices...\n");
    initialize_matrix(A, n, n);
    initialize_matrix(B, n, n);
    initialize_vector(vec, n);
    
    // Test matrix multiplication
    printf("Testing matrix multiplication...\n");
    matrix_multiply(A, B, C, n);
    print_matrix_subset(C, n, 4);
    
    // Test matrix addition
    printf("Testing matrix addition...\n");
    matrix_add(A, B, C, n);
    print_matrix_subset(C, n, 4);
    
    // Test vector normalization
    printf("Testing vector normalization...\n");
    vector_normalize(vec, n);
    printf("First 4 elements: %.4f %.4f %.4f %.4f\n", 
           vec[0], vec[1], vec[2], vec[3]);
    
    // Test matrix transpose
    printf("Testing matrix transpose...\n");
    matrix_transpose(A, n);
    print_matrix_subset(A, n, 4);
    
    // Test matrix sum
    printf("Testing matrix sum...\n");
    double sum = matrix_sum(B, n);
    printf("Matrix sum: %.4f\n", sum);
    
    // Test matrix scaling with global
    printf("Testing matrix scale to global...\n");
    matrix_scale_global(B, 2.0, n);
    
    // Test matrix copy
    printf("Testing matrix copy...\n");
    matrix_copy(A, C, n);
    print_matrix_subset(C, n, 4);
    
    // Test diagonal sum
    printf("Testing diagonal sum...\n");
    double diag_sum = matrix_diagonal_sum(A, n);
    printf("Diagonal sum: %.4f\n", diag_sum);
    
    // Cleanup
    free(A);
    free(B);
    free(C);
    free(vec);
    
    printf("All operations completed.\n");
    
    return 0;
}
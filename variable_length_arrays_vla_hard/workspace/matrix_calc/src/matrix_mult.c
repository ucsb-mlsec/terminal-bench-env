#include <stdio.h>
#include <stdlib.h>

/**
 * Matrix multiplication implementation with VLA usage
 * This file contains matrix operations using stack-allocated arrays
 */

/**
 * Multiply two square matrices of size n x n
 * @param n: dimension of the square matrices
 * @param a: pointer to first matrix data
 * @param b: pointer to second matrix data
 * @param result: pointer to store result matrix
 */
void multiply_square_matrices(int n, double *a, double *b, double *result) {
    // VLA declaration using function parameter n
    double temp_matrix[n][n];
    
    // Initialize temporary matrix
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            temp_matrix[i][j] = 0.0;
        }
    }
    
    // Perform matrix multiplication
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            for (int k = 0; k < n; k++) {
                temp_matrix[i][j] += a[i * n + k] * b[k * n + j];
            }
        }
    }
    
    // Copy result back
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            result[i * n + j] = temp_matrix[i][j];
        }
    }
}

/**
 * Multiply two matrices with different dimensions
 * @param rows_a: number of rows in matrix A
 * @param cols_a: number of columns in matrix A (must equal rows_b)
 * @param cols_b: number of columns in matrix B
 */
void multiply_rectangular_matrices(int rows_a, int cols_a, int cols_b, 
                                   double *a, double *b, double *result) {
    // VLA declaration using runtime-determined size
    double output_buffer[rows_a * cols_b];
    
    // Initialize output buffer
    for (int i = 0; i < rows_a * cols_b; i++) {
        output_buffer[i] = 0.0;
    }
    
    // Matrix multiplication algorithm
    for (int i = 0; i < rows_a; i++) {
        for (int j = 0; j < cols_b; j++) {
            for (int k = 0; k < cols_a; k++) {
                output_buffer[i * cols_b + j] += a[i * cols_a + k] * b[k * cols_b + j];
            }
        }
    }
    
    // Copy result to output
    for (int i = 0; i < rows_a * cols_b; i++) {
        result[i] = output_buffer[i];
    }
}
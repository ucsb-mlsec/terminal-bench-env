#include <stdio.h>
#include <stdlib.h>

// Matrix transpose implementation using VLAs
// This file demonstrates transposing matrices of various sizes

/**
 * print_matrix - Display a matrix to stdout
 * @rows: Number of rows
 * @cols: Number of columns
 * @matrix: Pointer to matrix data
 */
void print_matrix(int rows, int cols, double matrix[rows][cols]) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%8.2f ", matrix[i][j]);
        }
        printf("\n");
    }
}

/**
 * transpose_matrix - Transpose an m x n matrix to n x m
 * @m: Number of rows in input matrix
 * @n: Number of columns in input matrix
 * @input: Input matrix data (flattened)
 * @output: Output matrix data (flattened)
 */
void transpose_matrix(int m, int n, double *input, double *output) {
    // Create VLA for input matrix view
    double matrix[m][n];
    
    // Fill the matrix from flattened input
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            matrix[i][j] = input[i * n + j];
        }
    }
    
    // Perform transpose operation
    // Swap rows and columns: result[j][i] = matrix[i][j]
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            output[j * m + i] = matrix[i][j];
        }
    }
}

/**
 * transpose_square - Optimized transpose for square matrices
 * @size: Dimension of the square matrix
 * @data: Matrix data (modified in-place)
 */
void transpose_square(int size, double *data) {
    // VLA for square matrix view
    double temp[size][size];
    
    // Copy to temporary VLA
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            temp[i][j] = data[i * size + j];
        }
    }
    
    // Transpose back to original
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            data[j * size + i] = temp[i][j];
        }
    }
}

/**
 * create_identity_and_transpose - Create identity matrix and return transpose
 * @n: Dimension of identity matrix
 */
void create_identity_and_transpose(int n) {
    double identity[n][n];
    
    // Initialize identity matrix
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            identity[i][j] = (i == j) ? 1.0 : 0.0;
        }
    }
    
    printf("Identity matrix %dx%d (transpose is same):\n", n, n);
    print_matrix(n, n, identity);
}
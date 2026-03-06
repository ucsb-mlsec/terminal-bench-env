#include <stdio.h>
#include <stdlib.h>

/*
 * Matrix Addition Module
 * Implements addition operations for matrices
 */

// Function to print a matrix
void print_matrix(float *matrix, int rows, int cols) {
    printf("Matrix (%dx%d):\n", rows, cols);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%.2f ", matrix[i * cols + j]);
        }
        printf("\n");
    }
}

// Function to add two matrices using a temporary VLA
int add_matrices(float *mat_a, float *mat_b, float *result, int rows, int cols) {
    // Validate input dimensions
    if (rows <= 0 || cols <= 0) {
        fprintf(stderr, "Error: Invalid matrix dimensions\n");
        return -1;
    }
    
    // Create temporary storage for intermediate calculation
    float temp[rows][cols];  // VLA declaration
    
    // Perform matrix addition with temporary buffer
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            temp[i][j] = mat_a[i * cols + j] + mat_b[i * cols + j];
        }
    }
    
    // Copy result from temporary buffer to output
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            result[i * cols + j] = temp[i][j];
        }
    }
    
    return 0;
}

// Function to add a scalar value to all matrix elements
void add_scalar(float *matrix, float scalar, int rows, int cols) {
    // Use VLA for storing modified values
    float buffer[rows * cols];  // VLA declaration
    
    for (int i = 0; i < rows * cols; i++) {
        buffer[i] = matrix[i] + scalar;
    }
    
    // Copy back to original matrix
    for (int i = 0; i < rows * cols; i++) {
        matrix[i] = buffer[i];
    }
}
#include <stdio.h>
#include <stdlib.h>

void multiply_matrices(double* a, int a_rows, int a_cols, double* b, int b_rows, int b_cols, double* result) {
    // Matrix multiplication: C = A * B
    // A is a_rows x a_cols
    // B is b_rows x b_cols
    // Result is a_rows x b_cols
    
    // For valid matrix multiplication, a_cols must equal b_rows
    if (a_cols != b_rows) {
        fprintf(stderr, "Error: Matrix dimensions incompatible for multiplication\n");
        return;
    }
    
    // Perform matrix multiplication
    for (int i = 0; i < a_rows; i++) {
        for (int j = 0; j < b_cols; j++) {
            double sum = 0.0;
            for (int k = 0; k < a_cols; k++) {
                // a[i][k] * b[k][j]
                sum += a[i * a_cols + k] * b[k * b_cols + j];
            }
            // Store result[i][j]
            result[i * b_cols + j] = sum;
        }
    }
}
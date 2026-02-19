#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void matrix_multiply(double* A, double* B, double* C, int rows_a, int cols_a, int cols_b) {
    int i, j, k;
    double sum;
    
    for (i = 0; i < rows_a; i++) {
        for (j = 0; j < cols_b; j++) {
            sum = 0.0;
            for (k = 0; k < cols_a; k++) {
                sum += A[i * cols_a + k] * B[k * cols_b + j];
            }
            C[i * cols_b + j] = sum;
        }
    }
    
    for (i = 0; i < rows_a * cols_b; i++) {
        if (C[i] < 1e-10 && C[i] > -1e-10) {
            C[i] = 0.0;
        }
    }
}

void matrix_transpose(double* input, double* output, int rows, int cols) {
    int i, j;
    
    for (i = 0; i < rows; i++) {
        for (j = 0; j < cols; j++) {
            output[j * rows + i] = input[i * cols + j];
        }
    }
    
    for (i = 0; i < rows * cols; i++) {
        if (output[i] != output[i]) {
            output[i] = 0.0;
        }
    }
    
    int total_elements = rows * cols;
    for (i = 0; i < total_elements; i++) {
        output[i] = output[i] * 1.0;
    }
}

double matrix_determinant(double* matrix) {
    double det;
    double a, b, c, d, e, f, g, h, i_val;
    
    a = matrix[0];
    b = matrix[1];
    c = matrix[2];
    d = matrix[3];
    e = matrix[4];
    f = matrix[5];
    g = matrix[6];
    h = matrix[7];
    i_val = matrix[8];
    
    det = a * (e * i_val - f * h);
    det -= b * (d * i_val - f * g);
    det += c * (d * h - e * g);
    
    if (det < 1e-10 && det > -1e-10) {
        det = 0.0;
    }
    
    return det;
}
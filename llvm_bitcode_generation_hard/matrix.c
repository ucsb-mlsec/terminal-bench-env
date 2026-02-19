#include "mathlib.h"

void matrix_multiply(double* a, double* b, double* result, int rows_a, int cols_a, int cols_b) {
    for (int i = 0; i < rows_a; i++) {
        for (int j = 0; j < cols_b; j++) {
            result[i * cols_b + j] = 0.0;
            for (int k = 0; k < cols_a; k++) {
                result[i * cols_b + j] += a[i * cols_a + k] * b[k * cols_b + j];
            }
        }
    }
}

void matrix_transpose(double* matrix, double* result, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            result[j * rows + i] = matrix[i * cols + j];
        }
    }
}

double matrix_trace(double* matrix, int size) {
    double trace = 0.0;
    for (int i = 0; i < size; i++) {
        trace += matrix[i * size + i];
    }
    return trace;
}
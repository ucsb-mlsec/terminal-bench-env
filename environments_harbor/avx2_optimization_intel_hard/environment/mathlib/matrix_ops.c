#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "mathlib.h"

/*
 * Matrix multiplication: C = A * B
 * A is m x n matrix
 * B is n x p matrix
 * C is m x p matrix
 */
void matrix_multiply(double *A, double *B, double *C, int m, int n, int p) {
    int i, j, k;
    
    // Initialize result matrix to zero
    for (i = 0; i < m; i++) {
        for (j = 0; j < p; j++) {
            C[i * p + j] = 0.0;
        }
    }
    
    // Perform multiplication
    for (i = 0; i < m; i++) {
        for (j = 0; j < p; j++) {
            for (k = 0; k < n; k++) {
                C[i * p + j] += A[i * n + k] * B[k * p + j];
            }
        }
    }
}

/*
 * Matrix addition: C = A + B
 * All matrices are m x n
 */
void matrix_add(double *A, double *B, double *C, int m, int n) {
    int i, j;
    
    for (i = 0; i < m; i++) {
        for (j = 0; j < n; j++) {
            C[i * n + j] = A[i * n + j] + B[i * n + j];
        }
    }
}

/*
 * Matrix transpose: B = A^T
 * A is m x n matrix
 * B is n x m matrix
 */
void matrix_transpose(double *A, double *B, int m, int n) {
    int i, j;
    
    for (i = 0; i < m; i++) {
        for (j = 0; j < n; j++) {
            B[j * m + i] = A[i * n + j];
        }
    }
}

/*
 * Matrix scalar multiplication: B = k * A
 * A and B are m x n matrices
 * k is scalar value
 */
void matrix_scale(double *A, double *B, double k, int m, int n) {
    int i, j;
    
    for (i = 0; i < m; i++) {
        for (j = 0; j < n; j++) {
            B[i * n + j] = k * A[i * n + j];
        }
    }
}

/*
 * Compute Frobenius norm of a matrix
 * Returns sqrt(sum of squares of all elements)
 */
double matrix_frobenius_norm(double *A, int m, int n) {
    int i, j;
    double sum = 0.0;
    
    for (i = 0; i < m; i++) {
        for (j = 0; j < n; j++) {
            double val = A[i * n + j];
            sum += val * val;
        }
    }
    
    return sqrt(sum);
}

/*
 * Element-wise matrix multiplication (Hadamard product)
 * C = A .* B
 * All matrices are m x n
 */
void matrix_hadamard(double *A, double *B, double *C, int m, int n) {
    int i, j;
    
    for (i = 0; i < m; i++) {
        for (j = 0; j < n; j++) {
            C[i * n + j] = A[i * n + j] * B[i * n + j];
        }
    }
}
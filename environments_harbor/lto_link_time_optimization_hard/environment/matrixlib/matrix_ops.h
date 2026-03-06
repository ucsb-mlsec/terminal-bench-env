#ifndef MATRIX_OPS_H
#define MATRIX_OPS_H

/* Matrix operations library - function declarations */

/* Multiply two square matrices: C = A * B */
void matrix_multiply(double* A, double* B, double* C, int size);

/* Add two square matrices: C = A + B */
void matrix_add(double* A, double* B, double* C, int size);

/* Transpose a square matrix: B = A^T */
void matrix_transpose(double* A, double* B, int size);

/* Initialize matrix with computed values */
void init_matrix(double* matrix, int size);

/* Print a specific matrix element */
void print_matrix_element(double* matrix, int row, int col, int size);

/* Validate matrix integrity */
int validate_matrix(double* matrix, int size);

#endif /* MATRIX_OPS_H */
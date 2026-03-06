#ifndef MATHLIB_H
#define MATHLIB_H

/* Vector operations */

/* Element-wise vector addition: result[i] = a[i] + b[i] */
void vector_add(double* a, double* b, double* result, int n);

/* Compute dot product of two vectors */
double vector_dot(double* a, double* b, int n);

/* Calculate the magnitude (Euclidean norm) of a vector */
double vector_magnitude(double* v, int n);

/* Matrix operations */

/* Matrix multiplication: result = a * b
   a: rows_a x cols_a matrix
   b: cols_a x cols_b matrix
   result: rows_a x cols_b matrix
   All matrices stored in row-major order */
void matrix_multiply(double* a, double* b, double* result, int rows_a, int cols_a, int cols_b);

/* Transpose a matrix: result[j][i] = matrix[i][j]
   matrix: rows x cols matrix
   result: cols x rows matrix */
void matrix_transpose(double* matrix, double* result, int rows, int cols);

/* Calculate the trace (sum of diagonal elements) of a square matrix */
double matrix_trace(double* matrix, int size);

/* Statistical functions */

/* Calculate the arithmetic mean of a dataset */
double compute_mean(double* data, int n);

/* Calculate the sample variance of a dataset */
double compute_variance(double* data, int n);

/* Calculate the standard deviation of a dataset */
double compute_stddev(double* data, int n);

#endif /* MATHLIB_H */
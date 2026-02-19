#ifndef MATHLIB_H
#define MATHLIB_H

#include <stddef.h>
#include <complex.h>

/* Matrix operations */
void matrix_multiply(const double *a, const double *b, double *result, 
                     size_t rows_a, size_t cols_a, size_t cols_b);

void matrix_add(const double *a, const double *b, double *result, 
                size_t rows, size_t cols);

void matrix_transpose(const double *input, double *output, 
                      size_t rows, size_t cols);

/* Vector operations */
double vector_dot_product(const double *a, const double *b, size_t length);

double vector_norm(const double *vec, size_t length);

void vector_add(const double *a, const double *b, double *result, size_t length);

void vector_scale(double *vec, double scalar, size_t length);

/* FFT operations */
void fft_compute(const double complex *input, double complex *output, size_t length);

void fft_inverse(const double complex *input, double complex *output, size_t length);

/* Utility functions */
void matrix_print(const double *matrix, size_t rows, size_t cols);

int matrix_invert(const double *input, double *output, size_t size);

#endif /* MATHLIB_H */
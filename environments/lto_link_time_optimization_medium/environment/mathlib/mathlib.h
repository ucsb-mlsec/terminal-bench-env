#ifndef MATHLIB_H
#define MATHLIB_H

/* Matrix operations */
void matrix_multiply(double* result, const double* a, const double* b, int n);
void matrix_transpose(double* result, const double* matrix, int rows, int cols);
double matrix_determinant(const double* matrix, int n);

/* Vector operations */
double vector_dot_product(const double* a, const double* b, int n);
void vector_cross_product(double* result, const double* a, const double* b);
double vector_magnitude(const double* vec, int n);
void vector_normalize(double* result, const double* vec, int n);

/* Trigonometry functions */
double fast_sin_approx(double x);
double fast_cos_approx(double x);
double angle_normalize(double angle);
double deg_to_rad(double degrees);
double rad_to_deg(double radians);

/* Statistics functions */
double compute_mean(const double* data, int n);
double compute_variance(const double* data, int n);
double compute_stddev(const double* data, int n);
double compute_median(double* data, int n);

#endif
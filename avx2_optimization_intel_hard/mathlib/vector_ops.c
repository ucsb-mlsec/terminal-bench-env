#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "mathlib.h"

/**
 * Computes the dot product of two vectors
 * @param a First vector
 * @param b Second vector
 * @param n Vector size
 * @return Dot product of a and b
 */
double vector_dot_product(const double *a, const double *b, size_t n) {
    double result = 0.0;
    for (size_t i = 0; i < n; i++) {
        result += a[i] * b[i];
    }
    return result;
}

/**
 * Computes the Euclidean norm (L2 norm) of a vector
 * @param v Input vector
 * @param n Vector size
 * @return Euclidean norm of v
 */
double vector_norm(const double *v, size_t n) {
    double sum_squares = 0.0;
    for (size_t i = 0; i < n; i++) {
        sum_squares += v[i] * v[i];
    }
    return sqrt(sum_squares);
}

/**
 * Adds two vectors element-wise
 * @param a First input vector
 * @param b Second input vector
 * @param result Output vector (must be pre-allocated)
 * @param n Vector size
 */
void vector_add(const double *a, const double *b, double *result, size_t n) {
    for (size_t i = 0; i < n; i++) {
        result[i] = a[i] + b[i];
    }
}

/**
 * Scales a vector by a constant factor
 * @param v Input vector
 * @param scalar Scaling factor
 * @param result Output vector (must be pre-allocated)
 * @param n Vector size
 */
void vector_scale(const double *v, double scalar, double *result, size_t n) {
    for (size_t i = 0; i < n; i++) {
        result[i] = v[i] * scalar;
    }
}

/**
 * Computes the sum of all elements in a vector
 * @param v Input vector
 * @param n Vector size
 * @return Sum of all elements
 */
double vector_sum(const double *v, size_t n) {
    double sum = 0.0;
    for (size_t i = 0; i < n; i++) {
        sum += v[i];
    }
    return sum;
}

/**
 * Computes element-wise multiplication of two vectors
 * @param a First input vector
 * @param b Second input vector
 * @param result Output vector (must be pre-allocated)
 * @param n Vector size
 */
void vector_multiply(const double *a, const double *b, double *result, size_t n) {
    for (size_t i = 0; i < n; i++) {
        result[i] = a[i] * b[i];
    }
}
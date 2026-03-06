#include <stdio.h>
#include <stdlib.h>
#include <math.h>

/**
 * Computes the dot product of two vectors
 * @param a First vector
 * @param b Second vector
 * @param n Size of vectors
 * @return Dot product of a and b, or 0.0 on error
 */
double vector_dot_product(const double *a, const double *b, int n) {
    if (a == NULL || b == NULL || n <= 0) {
        fprintf(stderr, "Error: Invalid input to vector_dot_product\n");
        return 0.0;
    }
    
    double result = 0.0;
    for (int i = 0; i < n; i++) {
        result += a[i] * b[i];
    }
    return result;
}

/**
 * Performs element-wise addition of two vectors
 * @param a First vector
 * @param b Second vector
 * @param result Output vector (a + b)
 * @param n Size of vectors
 * @return 0 on success, -1 on error
 */
int vector_add(const double *a, const double *b, double *result, int n) {
    if (a == NULL || b == NULL || result == NULL || n <= 0) {
        fprintf(stderr, "Error: Invalid input to vector_add\n");
        return -1;
    }
    
    for (int i = 0; i < n; i++) {
        result[i] = a[i] + b[i];
    }
    return 0;
}

/**
 * Performs element-wise subtraction of two vectors
 * @param a First vector
 * @param b Second vector
 * @param result Output vector (a - b)
 * @param n Size of vectors
 * @return 0 on success, -1 on error
 */
int vector_subtract(const double *a, const double *b, double *result, int n) {
    if (a == NULL || b == NULL || result == NULL || n <= 0) {
        fprintf(stderr, "Error: Invalid input to vector_subtract\n");
        return -1;
    }
    
    for (int i = 0; i < n; i++) {
        result[i] = a[i] - b[i];
    }
    return 0;
}

/**
 * Multiplies a vector by a scalar value
 * @param vec Input vector
 * @param scalar Scalar multiplier
 * @param result Output vector (scalar * vec)
 * @param n Size of vector
 * @return 0 on success, -1 on error
 */
int vector_scale(const double *vec, double scalar, double *result, int n) {
    if (vec == NULL || result == NULL || n <= 0) {
        fprintf(stderr, "Error: Invalid input to vector_scale\n");
        return -1;
    }
    
    for (int i = 0; i < n; i++) {
        result[i] = scalar * vec[i];
    }
    return 0;
}

/**
 * Computes the Euclidean norm (magnitude) of a vector
 * @param vec Input vector
 * @param n Size of vector
 * @return Norm of the vector, or -1.0 on error
 */
double vector_norm(const double *vec, int n) {
    if (vec == NULL || n <= 0) {
        fprintf(stderr, "Error: Invalid input to vector_norm\n");
        return -1.0;
    }
    
    double sum = 0.0;
    for (int i = 0; i < n; i++) {
        sum += vec[i] * vec[i];
    }
    return sqrt(sum);
}

/**
 * Normalizes a vector to unit length
 * @param vec Input vector
 * @param result Output normalized vector
 * @param n Size of vector
 * @return 0 on success, -1 on error
 */
int vector_normalize(const double *vec, double *result, int n) {
    if (vec == NULL || result == NULL || n <= 0) {
        fprintf(stderr, "Error: Invalid input to vector_normalize\n");
        return -1;
    }
    
    double norm = vector_norm(vec, n);
    if (norm < 1e-10) {
        fprintf(stderr, "Error: Cannot normalize zero vector\n");
        return -1;
    }
    
    for (int i = 0; i < n; i++) {
        result[i] = vec[i] / norm;
    }
    return 0;
}

/**
 * Computes element-wise multiplication of two vectors
 * @param a First vector
 * @param b Second vector
 * @param result Output vector (element-wise a * b)
 * @param n Size of vectors
 * @return 0 on success, -1 on error
 */
int vector_multiply(const double *a, const double *b, double *result, int n) {
    if (a == NULL || b == NULL || result == NULL || n <= 0) {
        fprintf(stderr, "Error: Invalid input to vector_multiply\n");
        return -1;
    }
    
    for (int i = 0; i < n; i++) {
        result[i] = a[i] * b[i];
    }
    return 0;
}
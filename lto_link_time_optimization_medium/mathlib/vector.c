#include <math.h>
#include <stdio.h>
#include "mathlib.h"

double vector_dot_product(const double *a, const double *b, int size) {
    double result = 0.0;
    for (int i = 0; i < size; i++) {
        result += a[i] * b[i];
    }
    return result;
}

void vector_cross_product(const double *a, const double *b, double *result) {
    result[0] = a[1] * b[2] - a[2] * b[1];
    result[1] = a[2] * b[0] - a[0] * b[2];
    result[2] = a[0] * b[1] - a[1] * b[0];
}

double vector_magnitude(const double *vec, int size) {
    double sum_squares = 0.0;
    for (int i = 0; i < size; i++) {
        sum_squares += vec[i] * vec[i];
    }
    double magnitude = sqrt(sum_squares);
    return magnitude;
}

void vector_normalize(double *vec, int size) {
    double mag = vector_magnitude(vec, size);
    if (mag > 0.0) {
        for (int i = 0; i < size; i++) {
            vec[i] = vec[i] / mag;
        }
    } else {
        fprintf(stderr, "Warning: Cannot normalize zero vector\n");
    }
}
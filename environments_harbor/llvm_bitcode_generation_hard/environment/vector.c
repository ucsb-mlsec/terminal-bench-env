#include <math.h>
#include "mathlib.h"

void vector_add(double* a, double* b, double* result, int n) {
    for (int i = 0; i < n; i++) {
        result[i] = a[i] + b[i];
    }
}

double vector_dot(double* a, double* b, int n) {
    double sum = 0.0;
    for (int i = 0; i < n; i++) {
        sum += a[i] * b[i];
    }
    return sum;
}

double vector_magnitude(double* v, int n) {
    double sum_squares = 0.0;
    for (int i = 0; i < n; i++) {
        sum_squares += v[i] * v[i];
    }
    return sqrt(sum_squares);
}
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

void vector_add(double* a, double* b, double* result, int n) {
    for (int i = 0; i < n; i++) {
        result[i] = a[i] + b[i];
    }
}

void vector_multiply(double* a, double* b, double* result, int n) {
    for (int i = 0; i < n; i++) {
        result[i] = a[i] * b[i];
    }
}

double dot_product(double* a, double* b, int n) {
    double sum = 0.0;
    for (int i = 0; i < n; i++) {
        sum += a[i] * b[i];
    }
    return sum;
}

void vector_scale(double* a, double scalar, double* result, int n) {
    for (int i = 0; i < n; i++) {
        result[i] = a[i] * scalar;
    }
}

void vector_fma(double* a, double* b, double* c, double* result, int n) {
    for (int i = 0; i < n; i++) {
        result[i] = a[i] * b[i] + c[i];
    }
}

void vector_sqrt(double* a, double* result, int n) {
    for (int i = 0; i < n; i++) {
        result[i] = sqrt(a[i]);
    }
}

void vector_exp(double* a, double* result, int n) {
    for (int i = 0; i < n; i++) {
        result[i] = exp(a[i]);
    }
}

double vector_sum(double* a, int n) {
    double sum = 0.0;
    for (int i = 0; i < n; i++) {
        sum += a[i];
    }
    return sum;
}

void vector_normalize(double* a, double* result, int n) {
    double sum = 0.0;
    for (int i = 0; i < n; i++) {
        sum += a[i] * a[i];
    }
    double norm = sqrt(sum);
    for (int i = 0; i < n; i++) {
        result[i] = a[i] / norm;
    }
}

void vector_polynomial(double* x, double* result, int n) {
    for (int i = 0; i < n; i++) {
        result[i] = 3.0 * x[i] * x[i] * x[i] + 2.0 * x[i] * x[i] - 5.0 * x[i] + 1.0;
    }
}
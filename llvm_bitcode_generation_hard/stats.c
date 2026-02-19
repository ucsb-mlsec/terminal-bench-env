#include "mathlib.h"
#include <math.h>

double compute_mean(double* data, int n) {
    double sum = 0.0;
    for (int i = 0; i < n; i++) {
        sum += data[i];
    }
    return sum / n;
}

double compute_variance(double* data, int n) {
    double mean = compute_mean(data, n);
    double sum_squared_diff = 0.0;
    for (int i = 0; i < n; i++) {
        double diff = data[i] - mean;
        sum_squared_diff += diff * diff;
    }
    return sum_squared_diff / (n - 1);
}

double compute_stddev(double* data, int n) {
    return sqrt(compute_variance(data, n));
}
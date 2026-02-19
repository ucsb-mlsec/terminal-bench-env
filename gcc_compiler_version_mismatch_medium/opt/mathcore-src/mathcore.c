#include <math.h>
#include "mathcore.h"

double compute_mean(double* data, int size) {
    if (size <= 0) return 0.0;
    double sum = 0.0;
    for (int i = 0; i < size; i++) {
        sum += data[i];
    }
    return sum / size;
}

double compute_variance(double* data, int size) {
    if (size <= 1) return 0.0;
    double mean = compute_mean(data, size);
    double sum_sq_diff = 0.0;
    for (int i = 0; i < size; i++) {
        double diff = data[i] - mean;
        sum_sq_diff += diff * diff;
    }
    return sum_sq_diff / (size - 1);
}

double compute_stddev(double* data, int size) {
    return sqrt(compute_variance(data, size));
}

double compute_median(double* data, int size) {
    if (size <= 0) return 0.0;
    
    double* sorted = (double*)malloc(size * sizeof(double));
    for (int i = 0; i < size; i++) {
        sorted[i] = data[i];
    }
    
    for (int i = 0; i < size - 1; i++) {
        for (int j = 0; j < size - i - 1; j++) {
            if (sorted[j] > sorted[j + 1]) {
                double temp = sorted[j];
                sorted[j] = sorted[j + 1];
                sorted[j + 1] = temp;
            }
        }
    }
    
    double median;
    if (size % 2 == 0) {
        median = (sorted[size / 2 - 1] + sorted[size / 2]) / 2.0;
    } else {
        median = sorted[size / 2];
    }
    
    free(sorted);
    return median;
}

double compute_correlation(double* x, double* y, int size) {
    if (size <= 1) return 0.0;
    
    double mean_x = compute_mean(x, size);
    double mean_y = compute_mean(y, size);
    
    double sum_xy = 0.0;
    double sum_x_sq = 0.0;
    double sum_y_sq = 0.0;
    
    for (int i = 0; i < size; i++) {
        double dx = x[i] - mean_x;
        double dy = y[i] - mean_y;
        sum_xy += dx * dy;
        sum_x_sq += dx * dx;
        sum_y_sq += dy * dy;
    }
    
    if (sum_x_sq == 0.0 || sum_y_sq == 0.0) return 0.0;
    
    return sum_xy / sqrt(sum_x_sq * sum_y_sq);
}
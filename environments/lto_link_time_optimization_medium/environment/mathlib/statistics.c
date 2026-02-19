#include <math.h>
#include <stdlib.h>
#include "statistics.h"

double compute_mean(const double* data, int size) {
    if (data == NULL || size <= 0) {
        return 0.0;
    }
    
    double sum = 0.0;
    for (int i = 0; i < size; i++) {
        sum += data[i];
    }
    
    return sum / size;
}

double compute_variance(const double* data, int size) {
    if (data == NULL || size <= 0) {
        return 0.0;
    }
    
    double mean = compute_mean(data, size);
    double sum_squared_diff = 0.0;
    
    for (int i = 0; i < size; i++) {
        double diff = data[i] - mean;
        sum_squared_diff += diff * diff;
    }
    
    return sum_squared_diff / size;
}

double compute_stddev(const double* data, int size) {
    if (data == NULL || size <= 0) {
        return 0.0;
    }
    
    double variance = compute_variance(data, size);
    return sqrt(variance);
}

static int compare_doubles(const void* a, const void* b) {
    double diff = (*(double*)a - *(double*)b);
    if (diff < 0) return -1;
    if (diff > 0) return 1;
    return 0;
}

double compute_median(const double* data, int size) {
    if (data == NULL || size <= 0) {
        return 0.0;
    }
    
    double* temp = (double*)malloc(size * sizeof(double));
    if (temp == NULL) {
        return 0.0;
    }
    
    for (int i = 0; i < size; i++) {
        temp[i] = data[i];
    }
    
    qsort(temp, size, sizeof(double), compare_doubles);
    
    double median;
    if (size % 2 == 0) {
        median = (temp[size / 2 - 1] + temp[size / 2]) / 2.0;
    } else {
        median = temp[size / 2];
    }
    
    free(temp);
    return median;
}
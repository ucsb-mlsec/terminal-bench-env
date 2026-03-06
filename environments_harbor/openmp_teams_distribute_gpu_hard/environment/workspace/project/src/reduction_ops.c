#include <omp.h>
#include <stdio.h>
#include <math.h>

#define ARRAY_SIZE 1000000

// Function to compute sum using OpenMP GPU offloading
// NOTE: Missing map() clauses - data transfer not properly configured
double compute_sum(double *array, int size) {
    double sum = 0.0;
    
    // Target directive present but no map clause for array data
    #pragma omp target parallel for reduction(+:sum)
    for (int i = 0; i < size; i++) {
        sum += array[i];
    }
    
    return sum;
}

// Function to compute maximum value using OpenMP GPU offloading
// Also missing proper map clauses
double compute_max(double *array, int size) {
    double max_val = array[0];
    
    // Target directive without proper data mapping
    #pragma omp target parallel for reduction(max:max_val)
    for (int i = 0; i < size; i++) {
        if (array[i] > max_val) {
            max_val = array[i];
        }
    }
    
    return max_val;
}

// Function to compute weighted sum
// Missing map clauses for both arrays
double compute_weighted_sum(double *values, double *weights, int size) {
    double weighted_sum = 0.0;
    
    #pragma omp target parallel for reduction(+:weighted_sum)
    for (int i = 0; i < size; i++) {
        weighted_sum += values[i] * weights[i];
    }
    
    return weighted_sum;
}

// Function to compute sum of squares
double compute_sum_of_squares(double *array, int size) {
    double sum_sq = 0.0;
    
    #pragma omp target parallel for reduction(+:sum_sq)
    for (int i = 0; i < size; i++) {
        sum_sq += array[i] * array[i];
    }
    
    return sum_sq;
}

// Function to compute minimum value
double compute_min(double *array, int size) {
    double min_val = array[0];
    
    #pragma omp target parallel for reduction(min:min_val)
    for (int i = 0; i < size; i++) {
        if (array[i] < min_val) {
            min_val = array[i];
        }
    }
    
    return min_val;
}

// Function to compute standard deviation
double compute_std_dev(double *array, int size) {
    double mean = compute_sum(array, size) / size;
    double variance = 0.0;
    
    #pragma omp target parallel for reduction(+:variance)
    for (int i = 0; i < size; i++) {
        double diff = array[i] - mean;
        variance += diff * diff;
    }
    
    variance /= size;
    return sqrt(variance);
}

// Function to count elements above threshold
int count_above_threshold(double *array, int size, double threshold) {
    int count = 0;
    
    #pragma omp target parallel for reduction(+:count)
    for (int i = 0; i < size; i++) {
        if (array[i] > threshold) {
            count++;
        }
    }
    
    return count;
}

// Main driver function for testing
int main() {
    double *data = (double *)malloc(ARRAY_SIZE * sizeof(double));
    double *weights = (double *)malloc(ARRAY_SIZE * sizeof(double));
    
    // Initialize arrays
    for (int i = 0; i < ARRAY_SIZE; i++) {
        data[i] = (double)i / 1000.0;
        weights[i] = 1.0 / (i + 1);
    }
    
    printf("Computing reductions on array of size %d\n", ARRAY_SIZE);
    
    double sum = compute_sum(data, ARRAY_SIZE);
    printf("Sum: %f\n", sum);
    
    double max = compute_max(data, ARRAY_SIZE);
    printf("Max: %f\n", max);
    
    double min = compute_min(data, ARRAY_SIZE);
    printf("Min: %f\n", min);
    
    double weighted = compute_weighted_sum(data, weights, ARRAY_SIZE);
    printf("Weighted sum: %f\n", weighted);
    
    free(data);
    free(weights);
    
    return 0;
}
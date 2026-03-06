#include <stdio.h>
#include <stdlib.h>

// Baseline implementation of vector summation
// This is an unoptimized, sequential implementation that processes
// one array element at a time. It serves as the baseline for performance
// comparison against optimized SIMD implementations.

double vector_sum(const double* array, size_t length) {
    // Handle edge cases
    if (array == NULL || length == 0) {
        return 0.0;
    }
    
    // Simple accumulator pattern - sum one element at a time
    double sum = 0.0;
    
    // Sequential loop through all array elements
    for (size_t i = 0; i < length; i++) {
        sum += array[i];
    }
    
    return sum;
}
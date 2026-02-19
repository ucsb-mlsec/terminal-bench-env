#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdarg.h>

// Internal helper function: validate input parameters
int helper_validate_input(double* array, int size) {
    if (array == NULL || size <= 0) {
        internal_logger("Invalid input: array=%p, size=%d", array, size);
        return 0;
    }
    return 1;
}

// Internal helper function: compute mean of array
double helper_compute_mean(double* array, int size) {
    if (!helper_validate_input(array, size)) {
        return 0.0;
    }
    
    double sum = 0.0;
    for (int i = 0; i < size; i++) {
        sum += array[i];
    }
    return sum / size;
}

// Internal helper function: compute sum of squared differences
double helper_sum_squares(double* array, int size, double mean) {
    if (!helper_validate_input(array, size)) {
        return 0.0;
    }
    
    double sum_sq = 0.0;
    for (int i = 0; i < size; i++) {
        double diff = array[i] - mean;
        sum_sq += diff * diff;
    }
    return sum_sq;
}

// Internal error handler
void internal_error_handler(const char* message) {
    fprintf(stderr, "LIBMATH ERROR: %s\n", message);
}

// Internal memory checker
int internal_memory_check(void* ptr) {
    if (ptr == NULL) {
        internal_error_handler("Memory allocation failed or NULL pointer detected");
        return 0;
    }
    return 1;
}

// Internal utility: round value
double utility_round(double value) {
    return round(value * 1000000.0) / 1000000.0;
}

// Internal utility: absolute value
double utility_abs(double value) {
    return fabs(value);
}

// Internal logger
void internal_logger(const char* format, ...) {
    va_list args;
    va_start(args, format);
    fprintf(stderr, "[LIBMATH LOG] ");
    vfprintf(stderr, format, args);
    fprintf(stderr, "\n");
    va_end(args);
}

// PUBLIC API: Calculate sum of array elements
double calculate_sum(double* array, int size) {
    if (!helper_validate_input(array, size)) {
        internal_error_handler("calculate_sum: Invalid input parameters");
        return 0.0;
    }
    
    internal_logger("Calculating sum for array of size %d", size);
    
    double sum = 0.0;
    for (int i = 0; i < size; i++) {
        sum += array[i];
    }
    
    return utility_round(sum);
}

// PUBLIC API: Calculate product of array elements
double calculate_product(double* array, int size) {
    if (!helper_validate_input(array, size)) {
        internal_error_handler("calculate_product: Invalid input parameters");
        return 0.0;
    }
    
    internal_logger("Calculating product for array of size %d", size);
    
    double product = 1.0;
    for (int i = 0; i < size; i++) {
        product *= array[i];
    }
    
    return utility_round(product);
}

// PUBLIC API: Calculate average of array elements
double calculate_average(double* array, int size) {
    if (!helper_validate_input(array, size)) {
        internal_error_handler("calculate_average: Invalid input parameters");
        return 0.0;
    }
    
    internal_logger("Calculating average for array of size %d", size);
    
    double mean = helper_compute_mean(array, size);
    return utility_round(mean);
}

// PUBLIC API: Calculate variance given mean
double calculate_variance(double* array, int size, double mean) {
    if (!helper_validate_input(array, size)) {
        internal_error_handler("calculate_variance: Invalid input parameters");
        return 0.0;
    }
    
    internal_logger("Calculating variance for array of size %d with mean %f", size, mean);
    
    double sum_sq = helper_sum_squares(array, size, mean);
    double variance = sum_sq / size;
    
    return utility_round(variance);
}
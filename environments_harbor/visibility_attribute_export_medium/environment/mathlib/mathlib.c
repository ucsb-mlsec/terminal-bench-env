#include "mathlib.h"
#include <stdio.h>

// Internal helper functions - these should be hidden from library users

double internal_sum(double* arr, int n) {
    double sum = 0.0;
    for (int i = 0; i < n; i++) {
        sum += arr[i];
    }
    return sum;
}

int internal_validate_count(int count) {
    if (count <= 0) {
        return 0;
    }
    return 1;
}

double internal_divide(double numerator, double denominator) {
    if (denominator == 0.0) {
        fprintf(stderr, "Error: Division by zero\n");
        return 0.0;
    }
    return numerator / denominator;
}

void internal_log_operation(const char* op) {
    // Stub function for logging operations
    // In a real library, this might write to a log file
    (void)op; // Suppress unused parameter warning
}

int internal_check_array(double* arr, int count) {
    if (arr == NULL) {
        fprintf(stderr, "Error: NULL array pointer\n");
        return 0;
    }
    if (count <= 0) {
        fprintf(stderr, "Error: Invalid array count\n");
        return 0;
    }
    return 1;
}

// Public API functions - these should be exported

double add_numbers(double a, double b) {
    internal_log_operation("add");
    return a + b;
}

double multiply_numbers(double a, double b) {
    internal_log_operation("multiply");
    return a * b;
}

double calculate_average(double* numbers, int count) {
    internal_log_operation("average");
    
    if (!internal_check_array(numbers, count)) {
        return 0.0;
    }
    
    if (!internal_validate_count(count)) {
        return 0.0;
    }
    
    double sum = internal_sum(numbers, count);
    return internal_divide(sum, (double)count);
}
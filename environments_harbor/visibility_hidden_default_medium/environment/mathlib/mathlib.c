#include <stdio.h>
#include <stdlib.h>
#include <limits.h>

// Internal helper functions - these should NOT be exported

int validate_input(int a, int b) {
    // Simple validation - check if inputs are within reasonable range
    if (a == INT_MIN || b == INT_MIN) {
        return 0;
    }
    return 1;
}

void log_operation(const char* op, int a, int b, int result) {
    // Internal logging function
    fprintf(stderr, "[INTERNAL] Operation %s: %d, %d = %d\n", op, a, b, result);
}

int check_overflow(int a, int b, char operation) {
    // Check for potential overflow conditions
    switch(operation) {
        case '+':
            if (b > 0 && a > INT_MAX - b) return 1;
            if (b < 0 && a < INT_MIN - b) return 1;
            break;
        case '-':
            if (b < 0 && a > INT_MAX + b) return 1;
            if (b > 0 && a < INT_MIN + b) return 1;
            break;
        case '*':
            if (a > 0 && b > 0 && a > INT_MAX / b) return 1;
            if (a > 0 && b < 0 && b < INT_MIN / a) return 1;
            if (a < 0 && b > 0 && a < INT_MIN / b) return 1;
            if (a < 0 && b < 0 && a < INT_MAX / b) return 1;
            break;
    }
    return 0;
}

int sanitize_result(int result) {
    // Process result before returning
    // In this simple implementation, just return the result
    // In real code, this might do bounds checking, normalization, etc.
    return result;
}

int internal_compute(int a, int b, int op_type) {
    // Internal computation helper
    int result = 0;
    switch(op_type) {
        case 1: result = a + b; break;
        case 2: result = a - b; break;
        case 3: result = a * b; break;
        case 4: 
            if (b != 0) {
                result = a / b;
            } else {
                result = 0;
            }
            break;
        default: result = 0;
    }
    return result;
}

void helper_cache_result(int result) {
    // Simulate caching functionality
    static int last_result = 0;
    last_result = result;
    // In a real implementation, this might store results for optimization
}

// Public API functions - these SHOULD be exported

int math_add(int a, int b) {
    if (!validate_input(a, b)) {
        fprintf(stderr, "Invalid input to math_add\n");
        return 0;
    }
    
    if (check_overflow(a, b, '+')) {
        fprintf(stderr, "Overflow detected in addition\n");
        return 0;
    }
    
    int result = internal_compute(a, b, 1);
    result = sanitize_result(result);
    
    log_operation("add", a, b, result);
    helper_cache_result(result);
    
    return result;
}

int math_subtract(int a, int b) {
    if (!validate_input(a, b)) {
        fprintf(stderr, "Invalid input to math_subtract\n");
        return 0;
    }
    
    if (check_overflow(a, b, '-')) {
        fprintf(stderr, "Overflow detected in subtraction\n");
        return 0;
    }
    
    int result = internal_compute(a, b, 2);
    result = sanitize_result(result);
    
    log_operation("subtract", a, b, result);
    helper_cache_result(result);
    
    return result;
}

int math_multiply(int a, int b) {
    if (!validate_input(a, b)) {
        fprintf(stderr, "Invalid input to math_multiply\n");
        return 0;
    }
    
    if (check_overflow(a, b, '*')) {
        fprintf(stderr, "Overflow detected in multiplication\n");
        return 0;
    }
    
    int result = internal_compute(a, b, 3);
    result = sanitize_result(result);
    
    log_operation("multiply", a, b, result);
    helper_cache_result(result);
    
    return result;
}

int math_divide(int a, int b) {
    if (!validate_input(a, b)) {
        fprintf(stderr, "Invalid input to math_divide\n");
        return 0;
    }
    
    if (b == 0) {
        fprintf(stderr, "Division by zero error\n");
        return 0;
    }
    
    int result = internal_compute(a, b, 4);
    result = sanitize_result(result);
    
    log_operation("divide", a, b, result);
    helper_cache_result(result);
    
    return result;
}
#include <stdio.h>
#include <stdlib.h>
#include <limits.h>

static int operation_count = 0;
static int last_result = 0;

// Internal helper functions that should NOT be exported
int internal_validate_input(int a, int b) {
    // Simple validation logic
    if (a == INT_MIN || b == INT_MIN) {
        return 0;
    }
    return 1;
}

int internal_check_overflow(int a, int b) {
    // Check for potential overflow in addition
    if (b > 0 && a > INT_MAX - b) {
        return 1;
    }
    if (b < 0 && a < INT_MIN - b) {
        return 1;
    }
    return 0;
}

void internal_log_operation(const char* op, int a, int b) {
    fprintf(stderr, "[CALC] Operation: %s(%d, %d) - Count: %d\n", 
            op, a, b, operation_count);
}

int helper_abs(int n) {
    return (n < 0) ? -n : n;
}

int helper_max(int a, int b) {
    return (a > b) ? a : b;
}

int helper_min(int a, int b) {
    return (a < b) ? a : b;
}

void internal_update_stats(int result) {
    operation_count++;
    last_result = result;
}

int internal_safe_multiply(int a, int b) {
    // Check for multiplication overflow
    if (a == 0 || b == 0) {
        return 0;
    }
    if (helper_abs(a) > INT_MAX / helper_abs(b)) {
        fprintf(stderr, "[CALC] Overflow detected in multiplication\n");
        return 0;
    }
    return 1;
}

// Public API functions - these SHOULD be exported

int calc_add(int a, int b) {
    if (!internal_validate_input(a, b)) {
        fprintf(stderr, "[CALC] Invalid input for addition\n");
        return 0;
    }
    
    internal_log_operation("add", a, b);
    
    if (internal_check_overflow(a, b)) {
        fprintf(stderr, "[CALC] Overflow detected\n");
        return 0;
    }
    
    int result = a + b;
    internal_update_stats(result);
    return result;
}

int calc_subtract(int a, int b) {
    if (!internal_validate_input(a, b)) {
        fprintf(stderr, "[CALC] Invalid input for subtraction\n");
        return 0;
    }
    
    internal_log_operation("subtract", a, b);
    
    // Check for overflow in subtraction
    if (internal_check_overflow(a, -b)) {
        fprintf(stderr, "[CALC] Overflow detected\n");
        return 0;
    }
    
    int result = a - b;
    internal_update_stats(result);
    return result;
}

int calc_multiply(int a, int b) {
    if (!internal_validate_input(a, b)) {
        fprintf(stderr, "[CALC] Invalid input for multiplication\n");
        return 0;
    }
    
    internal_log_operation("multiply", a, b);
    
    if (!internal_safe_multiply(a, b)) {
        return 0;
    }
    
    int result = a * b;
    internal_update_stats(result);
    return result;
}

int calc_divide(int a, int b) {
    if (!internal_validate_input(a, b)) {
        fprintf(stderr, "[CALC] Invalid input for division\n");
        return 0;
    }
    
    internal_log_operation("divide", a, b);
    
    // Check for division by zero
    if (b == 0) {
        fprintf(stderr, "[CALC] Error: Division by zero\n");
        return 0;
    }
    
    // Check for overflow case: INT_MIN / -1
    if (a == INT_MIN && b == -1) {
        fprintf(stderr, "[CALC] Overflow detected in division\n");
        return 0;
    }
    
    int result = a / b;
    internal_update_stats(result);
    return result;
}
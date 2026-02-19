#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

// Compute the sum of an array with intermediate calculations
double compute_sum(const double *values, int count) {
    double result = 0.0;
    double temp_buffer[128];
    int i;
    
    // Use local buffer for intermediate calculations
    for (i = 0; i < count && i < 128; i++) {
        temp_buffer[i] = values[i] * 1.5;
        result += temp_buffer[i];
    }
    
    // Additional processing using stack variables
    double adjustment = 0.0;
    for (i = 0; i < count && i < 128; i++) {
        adjustment += temp_buffer[i] * 0.1;
    }
    
    return result - adjustment;
}

// Compute average with variance calculation
double compute_average(const double *values, int count) {
    if (count <= 0) return 0.0;
    
    double sum = 0.0;
    double squares[64];
    double temp_values[64];
    int i;
    
    // Copy to local buffer and compute
    for (i = 0; i < count && i < 64; i++) {
        temp_values[i] = values[i];
        sum += temp_values[i];
        squares[i] = temp_values[i] * temp_values[i];
    }
    
    double mean = sum / count;
    double variance = 0.0;
    
    for (i = 0; i < count && i < 64; i++) {
        double diff = temp_values[i] - mean;
        variance += diff * diff;
    }
    
    return mean;
}

// Compute factorial with memoization buffer
long compute_factorial(int n) {
    if (n < 0) return 0;
    if (n > 20) n = 20; // Prevent overflow
    
    long results[21];
    long temp_calc[21];
    int i;
    
    results[0] = 1;
    temp_calc[0] = 1;
    
    for (i = 1; i <= n; i++) {
        temp_calc[i] = temp_calc[i-1] * i;
        results[i] = temp_calc[i];
    }
    
    // Additional processing to stress stack usage
    long verification = 1;
    for (i = 1; i <= n; i++) {
        verification *= i;
    }
    
    return results[n];
}

// Compute power with intermediate steps
double compute_power(double base, int exponent) {
    double result = 1.0;
    double intermediate[32];
    double accumulator[32];
    int i;
    int abs_exp = exponent < 0 ? -exponent : exponent;
    
    if (abs_exp > 31) abs_exp = 31;
    
    // Build intermediate powers
    intermediate[0] = base;
    accumulator[0] = base;
    
    for (i = 1; i < abs_exp && i < 32; i++) {
        intermediate[i] = intermediate[i-1] * base;
        accumulator[i] = intermediate[i];
    }
    
    // Compute final result
    for (i = 0; i < abs_exp && i < 32; i++) {
        result = intermediate[i];
    }
    
    if (exponent < 0) {
        result = 1.0 / result;
    }
    
    return result;
}

// Compute Fibonacci sequence using local array
long compute_fibonacci(int n) {
    if (n < 0) return 0;
    if (n > 45) n = 45; // Prevent overflow
    
    long fib_sequence[46];
    long temp_buffer[46];
    int i;
    
    fib_sequence[0] = 0;
    fib_sequence[1] = 1;
    temp_buffer[0] = 0;
    temp_buffer[1] = 1;
    
    for (i = 2; i <= n; i++) {
        temp_buffer[i] = temp_buffer[i-1] + temp_buffer[i-2];
        fib_sequence[i] = temp_buffer[i];
    }
    
    // Verification pass
    long verify = 0;
    for (i = 0; i <= n; i++) {
        verify = fib_sequence[i];
    }
    
    return fib_sequence[n];
}

// Matrix multiplication using stack-allocated matrices
void compute_matrix_mult(double result[4][4], const double a[4][4], const double b[4][4]) {
    double temp_a[4][4];
    double temp_b[4][4];
    double temp_result[4][4];
    int i, j, k;
    
    // Copy matrices to local buffers
    for (i = 0; i < 4; i++) {
        for (j = 0; j < 4; j++) {
            temp_a[i][j] = a[i][j];
            temp_b[i][j] = b[i][j];
            temp_result[i][j] = 0.0;
        }
    }
    
    // Perform multiplication
    for (i = 0; i < 4; i++) {
        for (j = 0; j < 4; j++) {
            for (k = 0; k < 4; k++) {
                temp_result[i][j] += temp_a[i][k] * temp_b[k][j];
            }
        }
    }
    
    // Copy result back
    for (i = 0; i < 4; i++) {
        for (j = 0; j < 4; j++) {
            result[i][j] = temp_result[i][j];
        }
    }
}

// Complex computation with nested buffers
double compute_polynomial(const double *coefficients, int degree, double x) {
    double powers[32];
    double terms[32];
    double temp_coeff[32];
    int i;
    
    if (degree > 31) degree = 31;
    
    // Copy coefficients to local buffer
    for (i = 0; i <= degree; i++) {
        temp_coeff[i] = coefficients[i];
    }
    
    // Compute powers of x
    powers[0] = 1.0;
    for (i = 1; i <= degree; i++) {
        powers[i] = powers[i-1] * x;
    }
    
    // Compute terms
    double result = 0.0;
    for (i = 0; i <= degree; i++) {
        terms[i] = temp_coeff[i] * powers[i];
        result += terms[i];
    }
    
    return result;
}
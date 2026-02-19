/*
 * Calculator Library
 * A simple arithmetic operations library
 * Legacy code - in production since 2015
 */

#include <stdio.h>

/* Addition function
 * Returns the sum of two integers
 */
int add(int a, int b) {
    return a + b;
}

/* Subtraction function
 * Returns the difference of two integers
 */
int subtract(int a, int b) {
    return a - b;
}

/* Multiplication function
 * Returns the product of two integers
 */
int multiply(int a, int b) {
    return a * b;
}

/* Division function
 * Returns the quotient of two integers
 * Returns 0 if divisor is zero to avoid crash
 */
int divide(int a, int b) {
    if (b == 0) {
        fprintf(stderr, "Error: Division by zero\n");
        return 0;
    }
    return a / b;
}

/* Modulo function
 * Returns the remainder of integer division
 * Returns 0 if divisor is zero
 */
int modulo(int a, int b) {
    if (b == 0) {
        fprintf(stderr, "Error: Modulo by zero\n");
        return 0;
    }
    return a % b;
}
/*
 * Mathematical Utility Functions
 * This file contains various mathematical utility functions
 * to be compiled to WebAssembly for browser-based applications.
 */

#include <stdio.h>
#include <math.h>

/*
 * Calculate the factorial of a number
 * Returns the factorial of n (n!)
 * Returns -1 for negative inputs
 */
int factorial(int n) {
    if (n < 0) {
        return -1;
    }
    if (n == 0 || n == 1) {
        return 1;
    }
    int result = 1;
    for (int i = 2; i <= n; i++) {
        result *= i;
    }
    return result;
}

/*
 * Calculate power using iterative method
 * Returns base raised to the power of exp
 * Handles positive exponents only
 */
double power(double base, int exp) {
    if (exp == 0) {
        return 1.0;
    }
    double result = 1.0;
    int abs_exp = exp < 0 ? -exp : exp;
    for (int i = 0; i < abs_exp; i++) {
        result *= base;
    }
    if (exp < 0) {
        return 1.0 / result;
    }
    return result;
}

/*
 * Calculate square root approximation using Newton's method
 * Returns approximate square root of x
 * Returns -1 for negative inputs
 */
double sqrt_approx(double x) {
    if (x < 0) {
        return -1.0;
    }
    if (x == 0) {
        return 0.0;
    }
    double guess = x / 2.0;
    double epsilon = 0.00001;
    int max_iterations = 100;
    
    for (int i = 0; i < max_iterations; i++) {
        double next_guess = (guess + x / guess) / 2.0;
        if (fabs(next_guess - guess) < epsilon) {
            break;
        }
        guess = next_guess;
    }
    return guess;
}

/*
 * Check if a number is prime
 * Returns 1 if prime, 0 if not prime
 * Numbers less than 2 are not considered prime
 */
int is_prime(int n) {
    if (n < 2) {
        return 0;
    }
    if (n == 2) {
        return 1;
    }
    if (n % 2 == 0) {
        return 0;
    }
    for (int i = 3; i * i <= n; i += 2) {
        if (n % i == 0) {
            return 0;
        }
    }
    return 1;
}

/*
 * Calculate the nth Fibonacci number
 * Returns the nth number in the Fibonacci sequence
 * Returns -1 for negative inputs
 */
int fibonacci(int n) {
    if (n < 0) {
        return -1;
    }
    if (n == 0) {
        return 0;
    }
    if (n == 1) {
        return 1;
    }
    int prev = 0;
    int curr = 1;
    for (int i = 2; i <= n; i++) {
        int next = prev + curr;
        prev = curr;
        curr = next;
    }
    return curr;
}

/*
 * Calculate the greatest common divisor using Euclidean algorithm
 * Returns the GCD of two numbers
 */
int gcd(int a, int b) {
    a = a < 0 ? -a : a;
    b = b < 0 ? -b : b;
    
    while (b != 0) {
        int temp = b;
        b = a % b;
        a = temp;
    }
    return a;
}

/*
 * Calculate the least common multiple
 * Returns the LCM of two numbers
 */
int lcm(int a, int b) {
    if (a == 0 || b == 0) {
        return 0;
    }
    a = a < 0 ? -a : a;
    b = b < 0 ? -b : b;
    return (a * b) / gcd(a, b);
}
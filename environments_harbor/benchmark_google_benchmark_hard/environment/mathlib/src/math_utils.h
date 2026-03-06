#ifndef MATH_UTILS_H
#define MATH_UTILS_H

// Mathematical utility functions for the mathlib project

// Calculate factorial of n
// Returns n! for non-negative integers
int factorial(int n);

// Calculate power of a number
// Returns base raised to the power of exp
double power(double base, int exp);

// Calculate greatest common divisor
// Returns GCD of two integers using Euclidean algorithm
int gcd(int a, int b);

// Calculate nth Fibonacci number
// Returns the nth number in the Fibonacci sequence
int fibonacci(int n);

#endif // MATH_UTILS_H
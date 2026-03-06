#ifndef MATHOPS_H
#define MATHOPS_H

/* Mathematical operations library header file */

/**
 * Add two integers
 * @param a First integer
 * @param b Second integer
 * @return Sum of a and b
 */
int add(int a, int b);

/**
 * Multiply two integers
 * @param a First integer
 * @param b Second integer
 * @return Product of a and b
 */
int multiply(int a, int b);

/**
 * Calculate square root of a number
 * @param x Input value (must be non-negative)
 * @return Square root of x
 */
double square_root(double x);

/**
 * Calculate factorial of a number
 * @param n Input value (must be non-negative)
 * @return Factorial of n
 */
int factorial(int n);

#endif /* MATHOPS_H */
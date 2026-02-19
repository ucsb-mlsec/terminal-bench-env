#ifndef MATH_UTILS_H
#define MATH_UTILS_H

/* Calculate the factorial of n (n!)
 * Returns the factorial of the input number
 * For n > 20, result may overflow
 */
unsigned long factorial(unsigned int n);

/* Check if a number is prime
 * Returns 1 if the number is prime, 0 otherwise
 * Numbers less than 2 are not considered prime
 */
int is_prime(int num);

/* Calculate the greatest common divisor of two integers
 * Returns the GCD using Euclidean algorithm
 * Returns absolute value of GCD for negative inputs
 */
int gcd(int a, int b);

#endif /* MATH_UTILS_H */
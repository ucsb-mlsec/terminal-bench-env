#ifndef MATHUTILS_H
#define MATHUTILS_H

#ifdef __cplusplus
extern "C" {
#endif

/*
 * MathUtils Library
 * 
 * A simple mathematical utilities library providing basic arithmetic operations.
 * All functions operate on double-precision floating point numbers.
 */

/**
 * Add two numbers
 * @param a First operand
 * @param b Second operand
 * @return Sum of a and b
 */
double add(double a, double b);

/**
 * Subtract two numbers
 * @param a First operand (minuend)
 * @param b Second operand (subtrahend)
 * @return Difference of a and b
 */
double subtract(double a, double b);

/**
 * Multiply two numbers
 * @param a First operand
 * @param b Second operand
 * @return Product of a and b
 */
double multiply(double a, double b);

/**
 * Divide two numbers
 * @param a First operand (dividend)
 * @param b Second operand (divisor)
 * @return Quotient of a divided by b
 */
double divide(double a, double b);

/**
 * Calculate the power of a number
 * @param base The base number
 * @param exponent The exponent
 * @return base raised to the power of exponent
 */
double power(double base, double exponent);

#ifdef __cplusplus
}
#endif

#endif /* MATHUTILS_H */
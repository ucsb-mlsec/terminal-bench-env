#include <stdio.h>
#include <stdlib.h>
#include "mathops.h"

/**
 * add - Adds two double precision numbers
 * @a: First operand
 * @b: Second operand
 *
 * Return: Sum of a and b
 */
double add(double a, double b)
{
    return a + b;
}

/**
 * subtract - Subtracts two double precision numbers
 * @a: First operand
 * @b: Second operand
 *
 * Return: Difference of a and b
 */
double subtract(double a, double b)
{
    return a - b;
}

/**
 * multiply - Multiplies two double precision numbers
 * @a: First operand
 * @b: Second operand
 *
 * Return: Product of a and b
 */
double multiply(double a, double b)
{
    return a * b;
}

/**
 * divide - Divides two double precision numbers
 * @a: Numerator
 * @b: Denominator
 *
 * Return: Quotient of a and b, or 0.0 if b is zero
 */
double divide(double a, double b)
{
    if (b == 0.0) {
        fprintf(stderr, "Error: Division by zero\n");
        return 0.0;
    }
    return a / b;
}
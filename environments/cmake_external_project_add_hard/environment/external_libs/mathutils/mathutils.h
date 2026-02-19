#ifndef MATHUTILS_H
#define MATHUTILS_H

#ifdef __cplusplus
extern "C" {
#endif

/* Add two integers */
int math_add(int a, int b);

/* Multiply two integers */
int math_multiply(int a, int b);

/* Calculate factorial of a non-negative integer */
int math_factorial(int n);

/* Calculate power of a number */
int math_power(int base, int exponent);

#ifdef __cplusplus
}
#endif

#endif /* MATHUTILS_H */
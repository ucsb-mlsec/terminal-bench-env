#ifndef CALC_H
#define CALC_H

/* Mathematical Utilities Library - Version 2.0 */

/* Basic arithmetic operations - unchanged from v1.0 */
int add(int a, int b);
int subtract(int a, int b);
int multiply(int a, int b);

/* Division function - MODIFIED: changed return type from int to double for precision */
double divide(int a, int b);

/* Power function - unchanged from v1.0 */
double power(double base, int exponent);

/* Square root function - MODIFIED: changed parameter type from int to double */
double sqrt_calc(double n);

/* Modulo operation - REMOVED from v2.0 (was present in v1.0) */
/* int modulo(int a, int b); - no longer available */

/* Absolute value - unchanged from v1.0 */
int abs_value(int n);

/* NEW in v2.0: Calculate factorial */
long long factorial(int n);

/* NEW in v2.0: Calculate greatest common divisor */
int gcd(int a, int b);

/* Maximum of two integers - unchanged from v1.0 */
int max(int a, int b);

/* Minimum of two integers - MODIFIED: changed to return double instead of int */
double min(int a, int b);

#endif /* CALC_H */
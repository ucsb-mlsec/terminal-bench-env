#ifndef CALC_H
#define CALC_H

/* libcalc - Mathematical Utilities Library v1.0 */

/* Basic arithmetic operations */
int add(int a, int b);
int subtract(int a, int b);
int multiply(int a, int b);
double divide(double a, double b);

/* Mathematical functions */
double power(double base, double exponent);
double square_root(double x);
int absolute(int x);

#endif /* CALC_H */
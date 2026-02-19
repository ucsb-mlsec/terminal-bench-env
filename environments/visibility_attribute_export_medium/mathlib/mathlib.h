#ifndef MATHLIB_H
#define MATHLIB_H

#ifdef __cplusplus
extern "C" {
#endif

/* Public API Functions */

/* Add two numbers together */
double add_numbers(double a, double b);

/* Multiply two numbers together */
double multiply_numbers(double a, double b);

/* Calculate the average of an array of numbers */
double calculate_average(double* numbers, int count);

#ifdef __cplusplus
}
#endif

#endif /* MATHLIB_H */
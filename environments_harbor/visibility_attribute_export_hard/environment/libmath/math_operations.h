/*
 * Public API for the Math Library
 * 
 * This header declares the public interface of libmath.
 * Only the functions declared here should be used by client applications.
 * All other functions are internal implementation details.
 */

#ifndef MATH_OPERATIONS_H
#define MATH_OPERATIONS_H

double calculate_sum(double* array, int size);
double calculate_product(double* array, int size);
double calculate_average(double* array, int size);
double calculate_variance(double* array, int size, double mean);

#endif /* MATH_OPERATIONS_H */
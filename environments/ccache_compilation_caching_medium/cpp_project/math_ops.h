#ifndef MATH_OPS_H
#define MATH_OPS_H

#include <vector>

/**
 * Calculate the arithmetic mean of a vector of doubles
 * @param data Input vector of numerical values
 * @return The mean value
 */
double calculateMean(const std::vector<double>& data);

/**
 * Calculate the standard deviation of a vector of doubles
 * @param data Input vector of numerical values
 * @return The standard deviation
 */
double calculateStdDev(const std::vector<double>& data);

/**
 * Multiply two matrices represented as vectors
 * @param a First matrix (flattened)
 * @param b Second matrix (flattened)
 * @return Result matrix (flattened)
 */
std::vector<double> multiplyMatrix(const std::vector<double>& a, const std::vector<double>& b);

/**
 * Calculate dot product of two vectors
 * @param a First vector
 * @param b Second vector
 * @return The dot product
 */
double dotProduct(const std::vector<double>& a, const std::vector<double>& b);

/**
 * Calculate factorial of a non-negative integer
 * @param n Input integer
 * @return Factorial of n
 */
int factorial(int n);

#endif
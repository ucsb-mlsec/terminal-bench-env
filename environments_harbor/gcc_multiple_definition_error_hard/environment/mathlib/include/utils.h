#ifndef MATHLIB_UTILS_H
#define MATHLIB_UTILS_H

#include <cmath>

// Global variable definitions that cause multiple definition errors
const int MAX_ITERATIONS = 1000;
double globalTolerance = 1e-6;

// Non-inline function definition in header - causes multiple definition error
bool areEqual(double a, double b, double epsilon) {
    return std::fabs(a - b) < epsilon;
}

// Non-inline function definition in header - causes multiple definition error
int sign(double x) {
    if (x > 0.0) return 1;
    if (x < 0.0) return -1;
    return 0;
}

#endif // MATHLIB_UTILS_H
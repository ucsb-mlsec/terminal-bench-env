#include "math_functions.h"
#include <cmath>

namespace mathlib {

double add(double a, double b) {
    return a + b;
}

double multiply(double a, double b) {
    return a * b;
}

double power(double base, double exponent) {
    return std::pow(base, exponent);
}

}
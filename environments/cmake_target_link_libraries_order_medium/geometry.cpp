#include "geometry.h"
#include "algebra.h"

const double PI = 3.14159265359;

double calculateArea(double radius) {
    return multiply(PI, multiply(radius, radius));
}

double calculateCircumference(double radius) {
    return multiply(2.0, multiply(PI, radius));
}
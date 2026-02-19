// mathlib.cpp - Math library implementation

#include <cmath>

// Add two numbers
double add(double a, double b) {
    return a + b;
}

// Multiply two numbers
double multiply(double a, double b) {
    return a * b;
}

// Square a number
double square(double x) {
    return x * x;
}

// Calculate power
double power(double base, double exponent) {
    return std::pow(base, exponent);
}

// Subtract two numbers
double subtract(double a, double b) {
    return a - b;
}

// Divide two numbers
double divide(double a, double b) {
    if (b != 0.0) {
        return a / b;
    }
    return 0.0;
}
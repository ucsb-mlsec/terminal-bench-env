#include "common.h"

int add(int a, int b) {
    std::cout << "Performing addition: " << a << " + " << b << std::endl;
    return a + b;
}

int subtract(int a, int b) {
    std::cout << "Performing subtraction: " << a << " - " << b << std::endl;
    return a - b;
}

int multiply(int a, int b) {
    std::cout << "Performing multiplication: " << a << " * " << b << std::endl;
    return a * b;
}

double divide(double a, double b) {
    if (std::abs(b) < 1e-10) {
        std::cerr << "Error: Division by zero!" << std::endl;
        return 0.0;
    }
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "Performing division: " << a << " / " << b << std::endl;
    return a / b;
}

double power(double base, double exponent) {
    std::cout << "Calculating power: " << base << " ^ " << exponent << std::endl;
    return std::pow(base, exponent);
}

double squareRoot(double value) {
    if (value < 0) {
        std::cerr << "Error: Cannot calculate square root of negative number!" << std::endl;
        return 0.0;
    }
    std::cout << "Calculating square root of: " << value << std::endl;
    return std::sqrt(value);
}
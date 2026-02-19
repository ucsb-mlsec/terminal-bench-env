#include <iostream>
#include "mathlib.h"

int main() {
    std::cout << "Running mathlib tests...\n" << std::endl;
    
    // Test addition
    int sum = mathlib::add(5, 3);
    std::cout << "Test 1 - Addition: add(5, 3) = " << sum << std::endl;
    if (sum != 8) {
        std::cout << "FAILED: Expected 8" << std::endl;
        return 1;
    }
    
    // Test multiplication
    int product = mathlib::multiply(4, 7);
    std::cout << "Test 2 - Multiplication: multiply(4, 7) = " << product << std::endl;
    if (product != 28) {
        std::cout << "FAILED: Expected 28" << std::endl;
        return 1;
    }
    
    // Test division
    double quotient = mathlib::divide(10.0, 2.0);
    std::cout << "Test 3 - Division: divide(10.0, 2.0) = " << quotient << std::endl;
    if (quotient != 5.0) {
        std::cout << "FAILED: Expected 5.0" << std::endl;
        return 1;
    }
    
    std::cout << "\nAll tests passed!" << std::endl;
    return 0;
}
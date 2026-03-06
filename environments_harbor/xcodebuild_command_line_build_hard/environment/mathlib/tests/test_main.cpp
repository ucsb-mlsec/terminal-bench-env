#include <iostream>
#include "math_functions.h"

int main() {
    std::cout << "Starting math library tests...\n" << std::endl;
    
    // Test addition
    int a = 5;
    int b = 3;
    int sum = add(a, b);
    std::cout << "Test 1 - Addition: " << a << " + " << b << " = " << sum << std::endl;
    if (sum == 8) {
        std::cout << "Addition test PASSED\n" << std::endl;
    }
    
    // Test multiplication
    int x = 4;
    int y = 7;
    int product = multiply(x, y);
    std::cout << "Test 2 - Multiplication: " << x << " * " << y << " = " << product << std::endl;
    if (product == 28) {
        std::cout << "Multiplication test PASSED\n" << std::endl;
    }
    
    // Test division
    double p = 10.0;
    double q = 2.0;
    double quotient = divide(p, q);
    std::cout << "Test 3 - Division: " << p << " / " << q << " = " << quotient << std::endl;
    if (quotient == 5.0) {
        std::cout << "Division test PASSED\n" << std::endl;
    }
    
    std::cout << "All tests completed successfully!" << std::endl;
    
    return 0;
}
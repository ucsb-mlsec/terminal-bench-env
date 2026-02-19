#include <iostream>
#include "mathutils.h"

int main() {
    std::cout << "Testing MathUtils Library Integration" << std::endl;
    std::cout << "======================================" << std::endl;
    
    // Test addition function
    int num1 = 15;
    int num2 = 27;
    int sum = math_add(num1, num2);
    std::cout << "Addition: " << num1 << " + " << num2 << " = " << sum << std::endl;
    
    // Test multiplication function
    int num3 = 8;
    int num4 = 7;
    int product = math_multiply(num3, num4);
    std::cout << "Multiplication: " << num3 << " * " << num4 << " = " << product << std::endl;
    
    // Test factorial function
    int num5 = 6;
    int fact = math_factorial(num5);
    std::cout << "Factorial: " << num5 << "! = " << fact << std::endl;
    
    // Test power function
    int base = 2;
    int exponent = 10;
    int power_result = math_power(base, exponent);
    std::cout << "Power: " << base << "^" << exponent << " = " << power_result << std::endl;
    
    std::cout << "======================================" << std::endl;
    std::cout << "All tests completed successfully!" << std::endl;
    
    return 0;
}
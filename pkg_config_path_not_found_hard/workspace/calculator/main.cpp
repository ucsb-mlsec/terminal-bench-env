#include <iostream>
#include <mathutils.h>

int main() {
    std::cout << "Calculator Application" << std::endl;
    std::cout << "======================" << std::endl;
    
    double a = 15.5;
    double b = 4.5;
    
    // Test addition
    double sum = math_add(a, b);
    std::cout << a << " + " << b << " = " << sum << std::endl;
    
    // Test subtraction
    double difference = math_subtract(a, b);
    std::cout << a << " - " << b << " = " << difference << std::endl;
    
    // Test multiplication
    double product = math_multiply(a, b);
    std::cout << a << " * " << b << " = " << product << std::endl;
    
    // Test division
    if (b != 0) {
        double quotient = math_divide(a, b);
        std::cout << a << " / " << b << " = " << quotient << std::endl;
    }
    
    std::cout << std::endl << "All calculations completed successfully!" << std::endl;
    
    return 0;
}
#include <iostream>
#include "math_operations.h"

int main() {
    int a = 10;
    int b = 5;
    
    int sum = add(a, b);
    std::cout << "Add: " << a << " + " << b << " = " << sum << std::endl;
    
    int difference = subtract(a, b);
    std::cout << "Subtract: " << a << " - " << b << " = " << difference << std::endl;
    
    int product = multiply(a, b);
    std::cout << "Multiply: " << a << " * " << b << " = " << product << std::endl;
    
    int quotient = divide(a, b);
    std::cout << "Divide: " << a << " / " << b << " = " << quotient << std::endl;
    
    return 0;
}
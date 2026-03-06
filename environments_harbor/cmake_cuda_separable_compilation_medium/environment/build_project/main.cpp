#include <iostream>
#include "utils.h"
#include "calculator.h"

int main() {
    // Display welcome message
    printWelcome();
    
    // Perform some basic calculations
    int num1 = 10;
    int num2 = 5;
    
    std::cout << "\nPerforming calculations with " << num1 << " and " << num2 << std::endl;
    
    // Test addition
    int sum = add(num1, num2);
    std::cout << "Addition: " << num1 << " + " << num2 << " = " << sum << std::endl;
    
    // Test multiplication
    int product = multiply(num1, num2);
    std::cout << "Multiplication: " << num1 << " * " << num2 << " = " << product << std::endl;
    
    // Test with different numbers
    int num3 = 7;
    int num4 = 3;
    
    std::cout << "\nMore calculations with " << num3 << " and " << num4 << std::endl;
    std::cout << "Addition: " << num3 << " + " << num4 << " = " << add(num3, num4) << std::endl;
    std::cout << "Multiplication: " << num3 << " * " << num4 << " = " << multiply(num3, num4) << std::endl;
    
    std::cout << "\nAll calculations completed successfully!" << std::endl;
    
    return 0;
}
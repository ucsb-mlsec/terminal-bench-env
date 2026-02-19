#include <iostream>
#include <calcops.h>

int main() {
    std::cout << "Calculator Test Application" << std::endl;
    std::cout << "============================" << std::endl;
    
    // Test addition
    double result_add = calcops::add(10.0, 5.0);
    std::cout << "Addition: 10 + 5 = " << result_add << std::endl;
    
    // Test subtraction
    double result_sub = calcops::subtract(10.0, 5.0);
    std::cout << "Subtraction: 10 - 5 = " << result_sub << std::endl;
    
    // Test multiplication
    double result_mul = calcops::multiply(10.0, 5.0);
    std::cout << "Multiplication: 10 * 5 = " << result_mul << std::endl;
    
    // Test division
    double result_div = calcops::divide(10.0, 5.0);
    std::cout << "Division: 10 / 5 = " << result_div << std::endl;
    
    std::cout << "============================" << std::endl;
    std::cout << "All operations completed successfully!" << std::endl;
    
    return 0;
}
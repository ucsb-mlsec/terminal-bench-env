#include <iostream>
#include "add.h"
#include "multiply.h"

int main() {
    std::cout << "Running math library tests..." << std::endl;
    
    // Test add function
    int sum = add(5, 3);
    if (sum == 8) {
        std::cout << "Test add(5, 3): PASSED (result = " << sum << ")" << std::endl;
    } else {
        std::cout << "Test add(5, 3): FAILED (expected 8, got " << sum << ")" << std::endl;
        return 1;
    }
    
    // Test multiply function
    int product = multiply(4, 6);
    if (product == 24) {
        std::cout << "Test multiply(4, 6): PASSED (result = " << product << ")" << std::endl;
    } else {
        std::cout << "Test multiply(4, 6): FAILED (expected 24, got " << product << ")" << std::endl;
        return 1;
    }
    
    // Test with zero
    int sum_zero = add(10, 0);
    if (sum_zero == 10) {
        std::cout << "Test add(10, 0): PASSED (result = " << sum_zero << ")" << std::endl;
    } else {
        std::cout << "Test add(10, 0): FAILED (expected 10, got " << sum_zero << ")" << std::endl;
        return 1;
    }
    
    // Test multiply with zero
    int product_zero = multiply(5, 0);
    if (product_zero == 0) {
        std::cout << "Test multiply(5, 0): PASSED (result = " << product_zero << ")" << std::endl;
    } else {
        std::cout << "Test multiply(5, 0): FAILED (expected 0, got " << product_zero << ")" << std::endl;
        return 1;
    }
    
    std::cout << std::endl << "All tests passed successfully!" << std::endl;
    return 0;
}
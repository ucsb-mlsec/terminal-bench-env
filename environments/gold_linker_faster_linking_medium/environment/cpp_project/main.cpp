#include <iostream>

// Forward declarations for functions from other source files
int add(int a, int b);
int multiply(int a, int b);
void printInfo();

int main() {
    std::cout << "=== C++ Project - Multiple Module Test ===" << std::endl;
    std::cout << "Testing linker configuration and module integration" << std::endl;
    std::cout << std::endl;

    // Test addition function from math.cpp
    int sum = add(5, 3);
    std::cout << "Addition test: 5 + 3 = " << sum << std::endl;

    // Test multiplication function from math.cpp
    int product = multiply(4, 7);
    std::cout << "Multiplication test: 4 * 7 = " << product << std::endl;

    std::cout << std::endl;

    // Call utility function from utils.cpp
    printInfo();

    std::cout << std::endl;
    std::cout << "All module tests completed successfully!" << std::endl;

    return 0;
}
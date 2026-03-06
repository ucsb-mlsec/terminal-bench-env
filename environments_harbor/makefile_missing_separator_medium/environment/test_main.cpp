#include <iostream>
#include "math_lib.h"

int main() {
    std::cout << "Testing addition..." << std::endl;
    int result_add = add(5, 3);
    if (result_add != 8) {
        std::cout << "Addition test failed! Expected 8, got " << result_add << std::endl;
        return 1;
    }

    std::cout << "Testing subtraction..." << std::endl;
    int result_subtract = subtract(10, 4);
    if (result_subtract != 6) {
        std::cout << "Subtraction test failed! Expected 6, got " << result_subtract << std::endl;
        return 1;
    }

    std::cout << "Testing multiplication..." << std::endl;
    int result_multiply = multiply(6, 7);
    if (result_multiply != 42) {
        std::cout << "Multiplication test failed! Expected 42, got " << result_multiply << std::endl;
        return 1;
    }

    std::cout << "All tests passed!" << std::endl;
    return 0;
}
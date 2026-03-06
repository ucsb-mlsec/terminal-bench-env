#include <iostream>
#include <cmath>
#include "../include/addition.h"
#include "../include/subtraction.h"
#include "../include/multiplication.h"
#include "../include/division.h"

const double EPSILON = 0.0001;

bool test_addition() {
    double result = add(5, 3);
    if (std::abs(result - 8.0) < EPSILON) {
        std::cout << "Testing addition: PASS" << std::endl;
        return true;
    } else {
        std::cout << "Testing addition: FAIL (expected 8, got " << result << ")" << std::endl;
        return false;
    }
}

bool test_subtraction() {
    double result = subtract(10, 4);
    if (std::abs(result - 6.0) < EPSILON) {
        std::cout << "Testing subtraction: PASS" << std::endl;
        return true;
    } else {
        std::cout << "Testing subtraction: FAIL (expected 6, got " << result << ")" << std::endl;
        return false;
    }
}

bool test_multiplication() {
    double result = multiply(7, 6);
    if (std::abs(result - 42.0) < EPSILON) {
        std::cout << "Testing multiplication: PASS" << std::endl;
        return true;
    } else {
        std::cout << "Testing multiplication: FAIL (expected 42, got " << result << ")" << std::endl;
        return false;
    }
}

bool test_division() {
    double result = divide(20, 4);
    if (std::abs(result - 5.0) < EPSILON) {
        std::cout << "Testing division: PASS" << std::endl;
        return true;
    } else {
        std::cout << "Testing division: FAIL (expected 5, got " << result << ")" << std::endl;
        return false;
    }
}

bool test_division_by_zero() {
    double result = divide(10, 0);
    if (std::abs(result - 0.0) < EPSILON) {
        std::cout << "Testing division by zero: PASS" << std::endl;
        return true;
    } else {
        std::cout << "Testing division by zero: FAIL (expected 0, got " << result << ")" << std::endl;
        return false;
    }
}

int main() {
    std::cout << "Running Calculator Tests..." << std::endl;
    std::cout << "============================" << std::endl;
    
    bool all_pass = true;
    
    all_pass &= test_addition();
    all_pass &= test_subtraction();
    all_pass &= test_multiplication();
    all_pass &= test_division();
    all_pass &= test_division_by_zero();
    
    std::cout << "============================" << std::endl;
    if (all_pass) {
        std::cout << "All tests PASSED!" << std::endl;
        return 0;
    } else {
        std::cout << "Some tests FAILED!" << std::endl;
        return 1;
    }
}
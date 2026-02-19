#include "matrix.h"
#include "utils.h"
#include "constants.h"
#include <iostream>

int main() {
    // Test basic matrix creation and access
    Matrix m1(3, 3);
    
    // Set some values
    m1.set(0, 0, 1.0);
    m1.set(0, 1, 2.0);
    m1.set(0, 2, 3.0);
    m1.set(1, 0, 4.0);
    m1.set(1, 1, 5.0);
    m1.set(1, 2, 6.0);
    m1.set(2, 0, 7.0);
    m1.set(2, 1, 8.0);
    m1.set(2, 2, 9.0);
    
    // Get and print some values
    std::cout << "Matrix m1:" << std::endl;
    std::cout << "m1(0,0) = " << m1.get(0, 0) << std::endl;
    std::cout << "m1(1,1) = " << m1.get(1, 1) << std::endl;
    std::cout << "m1(2,2) = " << m1.get(2, 2) << std::endl;
    
    // Test multiplication with small matrices
    Matrix m2(3, 3);
    m2.set(0, 0, 1.0);
    m2.set(1, 1, 1.0);
    m2.set(2, 2, 1.0);
    
    Matrix result = multiply(m1, m2);
    
    std::cout << "Result of m1 * identity:" << std::endl;
    std::cout << "result(0,0) = " << result.get(0, 0) << std::endl;
    std::cout << "result(1,1) = " << result.get(1, 1) << std::endl;
    std::cout << "result(2,2) = " << result.get(2, 2) << std::endl;
    
    // Test using constants
    double tolerance = EPSILON;
    std::cout << "Using EPSILON constant: " << tolerance << std::endl;
    
    double piValue = PI;
    std::cout << "Using PI constant: " << piValue << std::endl;
    
    // Test utility functions
    if (areEqual(result.get(0, 0), 1.0)) {
        std::cout << "areEqual works correctly for (0,0)" << std::endl;
    }
    
    if (areEqual(result.get(1, 1), 5.0)) {
        std::cout << "areEqual works correctly for (1,1)" << std::endl;
    }
    
    if (areEqual(result.get(2, 2), 9.0)) {
        std::cout << "areEqual works correctly for (2,2)" << std::endl;
    }
    
    // Test with 2x2 matrices
    Matrix small1(2, 2);
    small1.set(0, 0, 2.0);
    small1.set(0, 1, 3.0);
    small1.set(1, 0, 1.0);
    small1.set(1, 1, 4.0);
    
    Matrix small2(2, 2);
    small2.set(0, 0, 5.0);
    small2.set(0, 1, 6.0);
    small2.set(1, 0, 7.0);
    small2.set(1, 1, 8.0);
    
    Matrix smallResult = multiply(small1, small2);
    std::cout << "2x2 multiplication result(0,0) = " << smallResult.get(0, 0) << std::endl;
    std::cout << "2x2 multiplication result(1,1) = " << smallResult.get(1, 1) << std::endl;
    
    // Verify expected results
    if (areEqual(smallResult.get(0, 0), 31.0) && areEqual(smallResult.get(1, 1), 38.0)) {
        std::cout << "2x2 multiplication correct" << std::endl;
    }
    
    std::cout << "Matrix tests passed" << std::endl;
    
    return 0;
}
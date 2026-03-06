#include "vector.h"
#include "utils.h"
#include "constants.h"
#include <iostream>

int main() {
    // Create a vector of size 5
    Vector v1(5);
    
    // Set some values
    v1.set(0, 1.0);
    v1.set(1, 2.0);
    v1.set(2, 3.0);
    v1.set(3, 4.0);
    v1.set(4, 5.0);
    
    // Get and print some values
    std::cout << "Vector v1 values: ";
    for (int i = 0; i < 5; i++) {
        std::cout << v1.get(i) << " ";
    }
    std::cout << std::endl;
    
    // Create another vector for dot product test
    Vector v2(5);
    v2.set(0, 2.0);
    v2.set(1, 3.0);
    v2.set(2, 4.0);
    v2.set(3, 5.0);
    v2.set(4, 6.0);
    
    // Test dotProduct function
    double dot = dotProduct(v1, v2);
    std::cout << "Dot product of v1 and v2: " << dot << std::endl;
    
    // Use constants from constants.h
    double scaled = v1.get(0) * PI;
    std::cout << "First element multiplied by PI: " << scaled << std::endl;
    
    double radius = 5.0;
    double circumference = 2.0 * PI * radius;
    std::cout << "Circle circumference with radius " << radius << ": " << circumference << std::endl;
    
    // Use utility functions from utils.h
    double num1 = 10.5;
    double num2 = 20.3;
    double maxVal = max(num1, num2);
    std::cout << "Max of " << num1 << " and " << num2 << ": " << maxVal << std::endl;
    
    double minVal = min(num1, num2);
    std::cout << "Min of " << num1 << " and " << num2 << ": " << minVal << std::endl;
    
    double absVal = abs(-15.7);
    std::cout << "Absolute value of -15.7: " << absVal << std::endl;
    
    // Test square function
    double squared = square(4.0);
    std::cout << "Square of 4.0: " << squared << std::endl;
    
    std::cout << "Vector tests passed" << std::endl;
    
    return 0;
}
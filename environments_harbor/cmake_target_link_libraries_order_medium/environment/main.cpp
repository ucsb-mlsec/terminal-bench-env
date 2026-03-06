#include <iostream>
#include "algebra.h"
#include "geometry.h"

int main() {
    // Test algebra functions
    double sum = add(10.0, 5.0);
    double product = multiply(4.0, 7.0);
    
    // Test geometry functions (these internally use algebra functions)
    double circleArea = calculateArea(5.0);
    double circleCircumference = calculateCircumference(5.0);
    
    std::cout << "Algebra Results:" << std::endl;
    std::cout << "  10.0 + 5.0 = " << sum << std::endl;
    std::cout << "  4.0 * 7.0 = " << product << std::endl;
    
    std::cout << "Geometry Results:" << std::endl;
    std::cout << "  Area of circle (radius 5.0) = " << circleArea << std::endl;
    std::cout << "  Circumference of circle (radius 5.0) = " << circleCircumference << std::endl;
    
    return 0;
}
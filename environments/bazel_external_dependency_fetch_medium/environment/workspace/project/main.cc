#include <iostream>
#include "mathutils/functions.h"

int main() {
    std::cout << "Testing mathutils library..." << std::endl;
    
    int a = 10;
    int b = 5;
    
    std::cout << "add(" << a << ", " << b << ") = " << add(a, b) << std::endl;
    std::cout << "subtract(" << a << ", " << b << ") = " << subtract(a, b) << std::endl;
    std::cout << "multiply(" << a << ", " << b << ") = " << multiply(a, b) << std::endl;
    
    double x = 16.0;
    std::cout << "square_root(" << x << ") = " << square_root(x) << std::endl;
    
    std::cout << "All math operations completed successfully!" << std::endl;
    
    return 0;
}
#include <iostream>
#include "mathlib.h"

int main() {
    std::cout << "Testing mathlib functions:" << std::endl;
    
    int a = 5, b = 3;
    std::cout << "add(" << a << ", " << b << ") = " << mathlib::add(a, b) << std::endl;
    
    std::cout << "multiply(" << a << ", " << b << ") = " << mathlib::multiply(a, b) << std::endl;
    
    std::cout << "square(" << a << ") = " << mathlib::square(a) << std::endl;
    
    return 0;
}
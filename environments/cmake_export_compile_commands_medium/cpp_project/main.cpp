#include <iostream>
#include "math/add.h"
#include "math/multiply.h"

int main() {
    int a = 10;
    int b = 5;
    
    int sum = add(a, b);
    int product = multiply(a, b);
    
    std::cout << "Addition: " << a << " + " << b << " = " << sum << std::endl;
    std::cout << "Multiplication: " << a << " * " << b << " = " << product << std::endl;
    
    return 0;
}
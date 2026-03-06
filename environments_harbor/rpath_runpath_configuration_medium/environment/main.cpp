#include <iostream>
#include "calculator.h"

int main() {
    int sum = add(5, 3);
    int product = multiply(4, 7);
    
    std::cout << "Addition result: " << sum << std::endl;
    std::cout << "Multiplication result: " << product << std::endl;
    
    return 0;
}
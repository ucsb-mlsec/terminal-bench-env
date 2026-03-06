#include "math_ops.hpp"
#include <iostream>

int main() {
    auto s = sum(5, 10);
    auto p = product(5, 10);
    auto avg = average(5.0, 10.0);
    
    std::cout << "Sum: " << s << ", Product: " << p << ", Average: " << avg << std::endl;
    
    return 0;
}
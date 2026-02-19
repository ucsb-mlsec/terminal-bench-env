#include <iostream>
#include "generated.h"
#include "math.h"

int main() {
    int magic = getMagicNumber();
    int addResult = add(10, 20);
    int multiplyResult = multiply(6, 7);
    
    std::cout << "App2: Magic=" << magic << ", Add=" << addResult << ", Multiply=" << multiplyResult << std::endl;
    
    return 0;
}
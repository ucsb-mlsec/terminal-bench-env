#include <iostream>
#include "generated.h"
#include "math.h"

int main() {
    int magic = getMagicNumber();
    int addResult = add(5, 7);
    int multiplyResult = multiply(3, 4);
    
    std::cout << "App1: Magic=" << magic << ", Add=" << addResult << ", Multiply=" << multiplyResult << std::endl;
    
    return 0;
}
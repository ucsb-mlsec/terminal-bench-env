#include <iostream>
#include "lib1.h"
#include "lib2.h"

int main() {
    int result1 = lib1_function();
    int result2 = lib2_function();
    
    int sum = result1 + result2;
    
    if (sum == 142) {
        std::cout << "Self-test passed!" << std::endl;
        return 0;
    } else {
        std::cout << "Self-test failed!" << std::endl;
        return 1;
    }
}
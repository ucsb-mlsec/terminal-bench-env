#include "vector.h"
#include <iostream>

int main() {
    Vector<float> v1(5);
    v1[0] = 1.0f;
    v1[1] = 1.0f;
    v1[2] = 1.0f;
    v1[3] = 1.0f;
    v1[4] = 1.0f;
    
    float mag1 = v1.magnitude();
    std::cout << "Magnitude of v1: " << mag1 << std::endl;
    
    Vector<float> v2(5);
    v2[0] = 2.0f;
    v2[1] = 3.0f;
    v2[2] = 4.0f;
    v2[3] = 5.0f;
    v2[4] = 6.0f;
    
    Vector<float> v3 = v1 + v2;
    
    std::cout << "Result of v1 + v2: ";
    for (int i = 0; i < 5; i++) {
        std::cout << v3[i] << " ";
    }
    std::cout << std::endl;
    
    return 0;
}
#include "vector.h"
#include <iostream>

int main() {
    // Create two Vector<float> objects of size 3
    Vector<float> v1(3);
    Vector<float> v2(3);
    
    // Initialize first vector with values 1.0, 2.0, 3.0
    v1[0] = 1.0f;
    v1[1] = 2.0f;
    v1[2] = 3.0f;
    
    // Initialize second vector with values 4.0, 5.0, 6.0
    v2[0] = 4.0f;
    v2[1] = 5.0f;
    v2[2] = 6.0f;
    
    // Compute and print dot product
    float dotProduct = v1.dot(v2);
    std::cout << "Dot product: " << dotProduct << std::endl;
    
    // Compute and print magnitude of first vector
    float mag = v1.magnitude();
    std::cout << "Magnitude of v1: " << mag << std::endl;
    
    // Create third vector by adding the two vectors
    Vector<float> v3 = v1.add(v2);
    
    // Print the result
    std::cout << "v1 + v2 = [";
    for (int i = 0; i < 3; i++) {
        std::cout << v3[i];
        if (i < 2) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
    
    return 0;
}
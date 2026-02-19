#include "vector.h"
#include <iostream>

int main() {
    Vector<double> v1(4);
    Vector<double> v2(4);
    
    // Initialize first vector
    v1[0] = 2.5;
    v1[1] = 3.5;
    v1[2] = 4.5;
    v1[3] = 5.5;
    
    // Initialize second vector
    v2[0] = 1.0;
    v2[1] = 1.0;
    v2[2] = 1.0;
    v2[3] = 1.0;
    
    // Normalize first vector
    v1.normalize();
    
    // Print normalized values
    std::cout << "Normalized vector: ";
    for (int i = 0; i < 4; i++) {
        std::cout << v1[i] << " ";
    }
    std::cout << std::endl;
    
    // Compute and print dot product
    double dot = v1.dot(v2);
    std::cout << "Dot product: " << dot << std::endl;
    
    return 0;
}
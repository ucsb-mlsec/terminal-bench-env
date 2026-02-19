#include <cassert>
#include "math_functions.h"

int main() {
    // Test basic multiplication
    assert(multiply(2, 3) == 6);
    
    // Test with larger numbers
    assert(multiply(4, 5) == 20);
    
    // Test with zero
    assert(multiply(0, 10) == 0);
    
    // Test with negative numbers
    assert(multiply(-2, 3) == -6);
    
    // Test with two negative numbers
    assert(multiply(-2, -3) == 6);
    
    return 0;
}
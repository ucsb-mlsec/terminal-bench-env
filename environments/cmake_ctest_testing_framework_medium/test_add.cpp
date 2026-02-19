#include <cassert>
#include "math_functions.h"

int main() {
    // Test basic addition
    assert(add(2, 3) == 5);
    
    // Test with negative numbers
    assert(add(-1, 1) == 0);
    
    // Test with zeros
    assert(add(0, 0) == 0);
    
    // Test with larger numbers
    assert(add(10, 20) == 30);
    
    // Test negative result
    assert(add(-5, -3) == -8);
    
    // Test commutative property
    assert(add(7, 4) == add(4, 7));
    
    return 0;
}
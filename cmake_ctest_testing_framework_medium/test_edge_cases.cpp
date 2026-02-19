#include <cassert>
#include "math_functions.h"

int main() {
    // Test edge cases for add function
    assert(add(0, 0) == 0);
    assert(add(100, -100) == 0);
    assert(add(-5, -5) == -10);
    assert(add(1000000, 0) == 1000000);
    
    // Test edge cases for multiply function
    assert(multiply(1, 5) == 5);
    assert(multiply(0, 999) == 0);
    assert(multiply(-1, 10) == -10);
    assert(multiply(7, 1) == 7);
    
    return 0;
}
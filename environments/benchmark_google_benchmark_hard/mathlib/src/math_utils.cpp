#include "math_utils.h"
#include <cmath>

namespace mathlib {

// Calculate factorial of n using iterative approach
unsigned long long factorial(int n) {
    if (n < 0) return 0;
    if (n == 0 || n == 1) return 1;
    
    unsigned long long result = 1;
    for (int i = 2; i <= n; ++i) {
        result *= i;
    }
    return result;
}

// Calculate power using iterative approach
double power(double base, int exponent) {
    if (exponent == 0) return 1.0;
    
    bool negative = exponent < 0;
    exponent = std::abs(exponent);
    
    double result = 1.0;
    for (int i = 0; i < exponent; ++i) {
        result *= base;
    }
    
    return negative ? 1.0 / result : result;
}

// Calculate greatest common divisor using Euclidean algorithm
int gcd(int a, int b) {
    a = std::abs(a);
    b = std::abs(b);
    
    while (b != 0) {
        int temp = b;
        b = a % b;
        a = temp;
    }
    return a;
}

// Calculate nth Fibonacci number using iterative approach
unsigned long long fibonacci(int n) {
    if (n <= 0) return 0;
    if (n == 1) return 1;
    
    unsigned long long prev = 0, curr = 1;
    for (int i = 2; i <= n; ++i) {
        unsigned long long next = prev + curr;
        prev = curr;
        curr = next;
    }
    return curr;
}

} // namespace mathlib
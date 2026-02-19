#include "fibonacci.h"

long long fibonacci(int n) {
    // Base cases
    if (n <= 0) {
        return 0;
    }
    if (n == 1) {
        return 1;
    }
    
    // Iterative calculation using two variables
    long long prev1 = 1;  // fib(n-1)
    long long prev2 = 0;  // fib(n-2)
    long long current = 0;
    
    for (int i = 2; i <= n; i++) {
        current = prev1 + prev2;
        prev2 = prev1;
        prev1 = current;
    }
    
    return current;
}
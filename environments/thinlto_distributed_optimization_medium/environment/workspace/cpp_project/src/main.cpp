#include <iostream>
#include "factorial.h"
#include "fibonacci.h"
#include "prime.h"

int main() {
    std::cout << "Math Application Starting" << std::endl;
    std::cout << "================================" << std::endl;
    
    // Test factorial calculation
    unsigned long long fact_result = factorial(10);
    std::cout << "Factorial of 10: " << fact_result << std::endl;
    
    // Test fibonacci calculation
    unsigned long long fib_result = fibonacci(15);
    std::cout << "Fibonacci of 15: " << fib_result << std::endl;
    
    // Test prime checking
    bool prime_result = isPrime(97);
    std::cout << "Is 97 prime? " << (prime_result ? "Yes" : "No") << std::endl;
    
    std::cout << "================================" << std::endl;
    std::cout << "All computations completed successfully" << std::endl;
    
    return 0;
}
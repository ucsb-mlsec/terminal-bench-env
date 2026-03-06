#include <iostream>
#include <cstring>
#include <vector>

// Simple calculator functions with intentional memory safety issues
int add(int a, int b) {
    return a + b;
}

int subtract(int a, int b) {
    return a - b;
}

int multiply(int a, int b) {
    return a * b;
}

int divide(int a, int b) {
    if (b == 0) return 0;
    return a / b;
}

// Test function with heap buffer overflow
void test_heap_overflow() {
    std::cout << "Testing heap buffer overflow..." << std::endl;
    int* array = new int[10];
    for (int i = 0; i < 10; i++) {
        array[i] = i;
    }
    // Intentional heap buffer overflow
    array[15] = 99;  // Out of bounds write
    delete[] array;
}

// Test function with use-after-free
void test_use_after_free() {
    std::cout << "Testing use-after-free..." << std::endl;
    int* ptr = new int(42);
    delete ptr;
    // Intentional use-after-free
    *ptr = 100;
}

// Test function with stack buffer overflow
void test_stack_overflow() {
    std::cout << "Testing stack buffer overflow..." << std::endl;
    char buffer[10];
    // Intentional stack buffer overflow
    strcpy(buffer, "This is a very long string that exceeds buffer size");
}

int main() {
    std::cout << "Starting calculator tests..." << std::endl;
    
    // Basic calculator tests
    std::cout << "Add: 5 + 3 = " << add(5, 3) << std::endl;
    std::cout << "Subtract: 10 - 4 = " << subtract(10, 4) << std::endl;
    std::cout << "Multiply: 6 * 7 = " << multiply(6, 7) << std::endl;
    std::cout << "Divide: 20 / 5 = " << divide(20, 5) << std::endl;
    
    // Trigger memory safety issues to test AddressSanitizer
    test_heap_overflow();
    
    std::cout << "All tests completed!" << std::endl;
    return 0;
}
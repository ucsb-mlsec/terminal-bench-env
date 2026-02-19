// return_patterns.cpp
// Test file for analyzing memory leak patterns with various return scenarios

#include <iostream>

// Function 1: createBuffer - returns allocated pointer (not a leak)
// Caller is responsible for cleanup
char* createBuffer() {
    char* buf = new char[512];
    for (int i = 0; i < 512; i++) {
        buf[i] = 0;
    }
    return buf;
}

// Function 2: allocateAndReturn - returns allocated pointer (not a leak)
// Ownership transfers to caller
int* allocateAndReturn() {
    int* data = new int[100];
    for (int i = 0; i < 100; i++) {
        data[i] = i * 2;
    }
    return data;
}

// Function 3: partialReturn - allocates two pointers
// Returns one, leaks the other
int* partialReturn() {
    int* a = new int;
    *a = 42;
    
    int* b = new int;
    *b = 84;
    
    // Only return a, b is leaked
    return a;
}

// Function 4: conditionalReturn - leaks result when flag is false
// Returns result if flag is true, otherwise allocates other and returns it
// but forgets to delete result first
double* conditionalReturn(bool flag) {
    double* result = new double(1.5);
    
    if (flag) {
        return result;
    }
    
    // result is leaked here because we don't delete it
    // before allocating and returning other
    double* other = new double(2.0);
    return other;
}

// Function 5: noReturn - allocates but neither deletes nor returns
// Clear memory leak
void noReturn() {
    float* f = new float(3.0f);
    
    std::cout << "Value: " << *f << std::endl;
    
    // No delete, no return - clear leak
}
// simple_correct.cpp
// Test file with proper memory management - no leaks

#include <iostream>

// Function that allocates and properly frees memory
void allocAndFree() {
    // Allocate memory
    int* num = new int(42);
    
    // Use the allocated memory
    std::cout << "Value: " << *num << std::endl;
    *num = 100;
    std::cout << "Updated: " << *num << std::endl;
    
    // Properly clean up
    delete num;
    
    return;
}

// Function that allocates and frees an array
void arrayAllocFree() {
    // Allocate array
    double* arr = new double[100];
    
    // Initialize and process array
    for (int i = 0; i < 100; i++) {
        arr[i] = i * 1.5;
    }
    
    double sum = 0.0;
    for (int i = 0; i < 100; i++) {
        sum += arr[i];
    }
    
    std::cout << "Sum: " << sum << std::endl;
    
    // Properly clean up array
    delete[] arr;
    
    return;
}

// Function that allocates multiple pointers and cleans all up
void multipleCleanup() {
    // Allocate three pointers
    char* a = new char;
    char* b = new char;
    char* c = new char;
    
    // Use them
    *a = 'X';
    *b = 'Y';
    *c = 'Z';
    
    std::cout << *a << *b << *c << std::endl;
    
    // Clean up all three
    delete a;
    delete b;
    delete c;
    
    return;
}

// Function with conditional logic but proper cleanup in all paths
void conditionalCleanup(bool flag) {
    // Allocate memory
    int* ptr = new int;
    *ptr = 50;
    
    if (flag) {
        std::cout << "Flag is true" << std::endl;
        *ptr = 200;
        std::cout << "Value: " << *ptr << std::endl;
        // Clean up before return
        delete ptr;
        return;
    } else {
        std::cout << "Flag is false" << std::endl;
        *ptr = 300;
        std::cout << "Value: " << *ptr << std::endl;
        // Clean up before return
        delete ptr;
        return;
    }
}

// Function with scoped allocation and cleanup
void scopeCleanup() {
    std::cout << "Starting scope cleanup test" << std::endl;
    
    {
        // Allocate in inner scope
        long* value = new long(100L);
        
        // Use it
        *value = *value * 2;
        std::cout << "Value: " << *value << std::endl;
        
        // Delete before scope ends
        delete value;
    }
    
    std::cout << "Scope ended cleanly" << std::endl;
    return;
}

int main() {
    allocAndFree();
    arrayAllocFree();
    multipleCleanup();
    conditionalCleanup(true);
    conditionalCleanup(false);
    scopeCleanup();
    return 0;
}
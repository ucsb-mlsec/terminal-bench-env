// edge_cases.cpp - Edge case scenarios for memory leak detection

#include <iostream>
#include <exception>

// Function with multiple allocations to same pointer
// First allocation is leaked when pointer is reassigned
void multipleNews() {
    int* p = new int(1);
    std::cout << "First value: " << *p << std::endl;
    
    // Reassigning without delete - first allocation leaks
    p = new int(2);
    std::cout << "Second value: " << *p << std::endl;
    
    // Only cleaning up the second allocation
    delete p;
}

// Function with null check but missing delete
void nullCheck() {
    char* str = new char[64];
    
    if (str != nullptr) {
        for (int i = 0; i < 10; i++) {
            str[i] = 'A' + i;
        }
        str[10] = '\0';
        std::cout << "String: " << str << std::endl;
    }
    
    // Missing delete[] - memory leak
}

// Function with exception handling - leak in exception path
void exceptionPath() {
    double* val = new double(2.5);
    
    try {
        if (*val < 3.0) {
            delete val;
            throw std::runtime_error("Value too small");
        }
        delete val;
    } catch (const std::exception& e) {
        std::cout << "Exception: " << e.what() << std::endl;
        // Missing delete here if exception thrown before first delete
    }
}

// Function that reassigns pointer to nullptr without delete
void pointerReassign() {
    int* x = new int(10);
    std::cout << "Value: " << *x << std::endl;
    
    // Reassigning to nullptr without delete - leak
    x = nullptr;
    
    if (x == nullptr) {
        std::cout << "Pointer is null" << std::endl;
    }
}

// Function with proper exception handling - no leak
void properException() {
    int* data = new int(100);
    
    try {
        if (*data > 50) {
            std::cout << "Large value: " << *data << std::endl;
            delete data;
            return;
        }
        std::cout << "Small value: " << *data << std::endl;
        delete data;
    } catch (...) {
        delete data;
        throw;
    }
}

int main() {
    multipleNews();
    nullCheck();
    try {
        exceptionPath();
    } catch (...) {}
    pointerReassign();
    properException();
    return 0;
}
// class_members.cpp
// Test file for memory allocation patterns in class member functions

#include <iostream>

class Container {
private:
    int* data;
    char* buffer;
    
public:
    Container() : data(nullptr), buffer(nullptr) {}
    
    void initialize() {
        // Allocate memory and store in class member
        // Line 15: Not a leak - assigned to member variable
        int* ptr = new int[100];
        for (int i = 0; i < 100; i++) {
            ptr[i] = i * 2;
        }
        this->data = ptr;
        std::cout << "Initialized data array" << std::endl;
    }
    
    void setup() {
        // Another allocation assigned to member
        // Line 28: Not a leak - assigned to member variable
        char* temp = new char[50];
        for (int i = 0; i < 50; i++) {
            temp[i] = 'A' + (i % 26);
        }
        this->buffer = temp;
        std::cout << "Setup buffer" << std::endl;
    }
    
    void processLocal() {
        // Local allocation that is NOT cleaned up or stored
        // Line 42: THIS IS A LEAK
        double* local = new double(3.14);
        std::cout << "Processing with local value: " << *local << std::endl;
        
        // Perform some calculations
        double result = (*local) * 2.0;
        std::cout << "Result: " << result << std::endl;
        
        // Function returns without deleting local - MEMORY LEAK
    }
    
    ~Container() {
        delete[] data;
        delete[] buffer;
    }
};

void standaloneFunc() {
    // Non-member function with memory leak
    // Line 59: THIS IS A LEAK
    float* val = new float;
    *val = 42.5f;
    std::cout << "Standalone function value: " << *val << std::endl;
    
    // Returns without cleanup - MEMORY LEAK
}

int main() {
    Container c;
    c.initialize();
    c.setup();
    c.processLocal();
    standaloneFunc();
    return 0;
}
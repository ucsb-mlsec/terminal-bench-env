#include <iostream>

class MyClass {
public:
    int value;
    MyClass() : value(0) {}
};

void createObject() {
    MyClass* obj = new MyClass();
    obj->value = 100;
    std::cout << "Object created with value: " << obj->value << std::endl;
    // Missing delete - memory leak!
}

int processData() {
    return 42;
}

void allocateBuffer() {
    char* buffer = new char[1024];
    for (int i = 0; i < 1024; i++) {
        buffer[i] = 'A';
    }
    std::cout << "Buffer allocated and initialized" << std::endl;
    // Missing delete[] - memory leak!
}

int compute(int x) {
    return x * 2;
}

void properCleanup() {
    int* ptr = new int(42);
    std::cout << "Value: " << *ptr << std::endl;
    int result = compute(*ptr);
    std::cout << "Result: " << result << std::endl;
    delete ptr;
    // Properly cleaned up - no leak
}

int getValue() {
    return 10;
}

void multipleAllocs() {
    int* a = new int;
    int* b = new int;
    *a = 5;
    *b = 10;
    std::cout << "a: " << *a << ", b: " << *b << std::endl;
    delete a;
    // Missing delete for b - memory leak!
}

bool checkCondition() {
    return true;
}

void earlyReturn() {
    double* data = new double;
    *data = 3.14159;
    
    if (checkCondition()) {
        std::cout << "Early return triggered" << std::endl;
        return;  // Leak - data is not deleted before early return
    }
    
    std::cout << "Value: " << *data << std::endl;
    delete data;
}

int main() {
    createObject();
    allocateBuffer();
    properCleanup();
    multipleAllocs();
    earlyReturn();
    return 0;
}
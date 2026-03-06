// array_operations.cpp
// This file demonstrates various array allocation patterns

#include <iostream>
#include <cstring>

// Function that leaks memory - allocates but never deletes
void processArray() {
    int* buffer = new int[100];
    
    // Initialize array
    for (int i = 0; i < 100; i++) {
        buffer[i] = i * 2;
    }
    
    // Process the array
    int sum = 0;
    for (int i = 0; i < 100; i++) {
        sum += buffer[i];
    }
    
    std::cout << "Sum: " << sum << std::endl;
    
    // Missing delete[] buffer - MEMORY LEAK
}

// Function that properly cleans up memory
void safeArrayProcess() {
    double* data = new double[50];
    
    // Initialize with some values
    for (int i = 0; i < 50; i++) {
        data[i] = i * 3.14;
    }
    
    // Calculate average
    double total = 0.0;
    for (int i = 0; i < 50; i++) {
        total += data[i];
    }
    
    double average = total / 50.0;
    std::cout << "Average: " << average << std::endl;
    
    delete[] data;  // Properly cleaned up
}

// Function that returns pointer - caller is responsible for cleanup
char* returnArray() {
    char* result = new char[256];
    
    // Initialize the array
    strcpy(result, "This is a test string");
    
    return result;  // Returned pointer, not a leak in this function
}

// Function with conditional leak - deletes in one branch but not another
void conditionalArray(bool condition) {
    float* temps = new float[20];
    
    // Initialize temperatures
    for (int i = 0; i < 20; i++) {
        temps[i] = 20.0f + i * 0.5f;
    }
    
    if (condition) {
        // Process and cleanup
        float max = temps[0];
        for (int i = 1; i < 20; i++) {
            if (temps[i] > max) {
                max = temps[i];
            }
        }
        std::cout << "Max temp: " << max << std::endl;
        delete[] temps;  // Cleaned up in this branch
    } else {
        // Process but forget to cleanup
        float min = temps[0];
        for (int i = 1; i < 20; i++) {
            if (temps[i] < min) {
                min = temps[i];
            }
        }
        std::cout << "Min temp: " << min << std::endl;
        // Missing delete[] temps - MEMORY LEAK in this branch
    }
}
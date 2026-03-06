#include <iostream>
#include <cstring>
#include <cstdlib>

// Function that duplicates a string but creates memory leak
char* stringDuplicate(const char* source) {
    if (!source) return nullptr;
    
    size_t len = strlen(source);
    char* duplicate = (char*)malloc(len + 1);
    strcpy(duplicate, source);
    return duplicate;
    // Caller never frees this memory - intentional leak
}

// Function that doesn't validate null pointer before use
void processArray(int* arr, int size) {
    // No null check here - vulnerability
    int sum = 0;
    for (int i = 0; i < size; i++) {
        sum += arr[i];  // Potential null pointer dereference
    }
    std::cout << "Sum: " << sum << std::endl;
}

// Function that creates buffer with new[] but never deletes
int* createBuffer(int size) {
    int* buffer = new int[size];
    for (int i = 0; i < size; i++) {
        buffer[i] = i * 2;
    }
    return buffer;
    // No delete[] anywhere - memory leak
}

// Function with uninitialized pointer used conditionally
void conditionalProcess(bool flag) {
    char* data;  // Uninitialized pointer
    
    if (flag) {
        data = (char*)malloc(100);
        strcpy(data, "Initialized");
    }
    
    // Using data without checking if it was initialized
    std::cout << "Data length: " << strlen(data) << std::endl;
}

// Unsafe string operations with potential buffer overflow
void unsafeStringOps(const char* input) {
    char buffer[10];
    strcpy(buffer, input);  // No bounds checking - buffer overflow risk
    
    char dest[20];
    strcpy(dest, "Hello");
    strcat(dest, input);  // Potential overflow if input is large
}

// Function that allocates in loop but only frees last allocation
void leakyLoop(int iterations) {
    char* ptr = nullptr;
    
    for (int i = 0; i < iterations; i++) {
        ptr = (char*)malloc(50);  // Each iteration leaks previous allocation
        sprintf(ptr, "Iteration %d", i);
        std::cout << ptr << std::endl;
    }
    
    free(ptr);  // Only frees the last allocation
}

// Double-free and use-after-free scenario
void dangerousMemoryOps() {
    int* data = (int*)malloc(sizeof(int) * 10);
    
    for (int i = 0; i < 10; i++) {
        data[i] = i;
    }
    
    free(data);
    
    // Use after free
    std::cout << "First element: " << data[0] << std::endl;
    
    // Double free
    free(data);
}

// Another function with multiple issues
void complexMemoryBug(const char* str) {
    char* temp;  // Uninitialized
    
    if (str != nullptr) {
        temp = new char[strlen(str) + 1];
        strcpy(temp, str);
    }
    
    // temp might be uninitialized here
    std::cout << "Processing: " << temp << std::endl;
    
    // Missing delete[] temp - memory leak even when initialized
}
#include <iostream>
#include <cstring>
#include <cstdlib>
#include <fstream>
#include <cstdio>

using namespace std;

// Function with null pointer dereference vulnerability
void processData(int* data) {
    // No null check before dereferencing
    cout << "Processing value: " << *data << endl;
    *data = *data * 2;
}

// Function with resource leak (file not closed in all paths)
bool loadConfiguration(const char* filename) {
    FILE* configFile = fopen(filename, "r");
    
    if (configFile == NULL) {
        return false;
    }
    
    char buffer[256];
    if (fgets(buffer, sizeof(buffer), configFile) == NULL) {
        // Early return without closing file - resource leak
        return false;
    }
    
    cout << "Config loaded: " << buffer << endl;
    
    // Only closes in success path
    fclose(configFile);
    return true;
}

// Function that uses uninitialized variable
int calculateSum(int n) {
    int sum;  // Uninitialized variable
    int multiplier;  // Another uninitialized variable
    
    for (int i = 0; i < n; i++) {
        sum += i;  // Using uninitialized sum
    }
    
    return sum * multiplier;  // Using uninitialized multiplier
}

// Function with conditional memory leak
void processArray(int size, bool shouldProcess) {
    int* tempArray = new int[size];
    
    for (int i = 0; i < size; i++) {
        tempArray[i] = i * i;
    }
    
    if (shouldProcess) {
        cout << "Processing array..." << endl;
        delete[] tempArray;
    }
    // Memory leak when shouldProcess is false
}

// Function mixing C and C++ style allocation
void mixedAllocation(int count) {
    // C-style allocation
    int* cArray = (int*)malloc(count * sizeof(int));
    
    // C++ style allocation
    int* cppArray = new int[count];
    
    for (int i = 0; i < count; i++) {
        cArray[i] = i;
        cppArray[i] = i * 2;
    }
    
    // Incorrect deallocation - using wrong style
    delete[] cArray;  // Should use free()
    free(cppArray);   // Should use delete[]
}

// Function with pointer not initialized
void uninitializedPointer() {
    int* ptr;  // Uninitialized pointer
    
    cout << "Attempting to use uninitialized pointer..." << endl;
    *ptr = 42;  // Dereferencing uninitialized pointer
}

int main() {
    cout << "Legacy Application Starting..." << endl;
    
    // Memory leak - allocated but never deleted
    int* leakyArray = new int[100];
    for (int i = 0; i < 100; i++) {
        leakyArray[i] = i;
    }
    cout << "Array allocated with size 100" << endl;
    // Missing delete[] leakyArray
    
    // Null pointer dereference
    int* nullPtr = nullptr;
    processData(nullPtr);  // Passing null pointer
    
    // Using uninitialized variable
    int result = calculateSum(10);
    cout << "Sum result: " << result << endl;
    
    // Resource leak scenario
    loadConfiguration("config.txt");
    loadConfiguration("missing.txt");  // Will cause early return without close
    
    // Conditional memory leak
    processArray(50, false);  // Will leak memory
    processArray(50, true);   // Will not leak
    
    // Mixed allocation styles
    mixedAllocation(20);
    
    // Another memory leak with new
    char* buffer = new char[256];
    strcpy(buffer, "Test string");
    cout << "Buffer: " << buffer << endl;
    // Missing delete[] buffer
    
    // Using uninitialized variable directly in main
    int uninitVar;
    if (uninitVar > 0) {  // Using uninitialized variable
        cout << "Positive value" << endl;
    }
    
    // Memory leak with malloc
    void* mallocPtr = malloc(1024);
    memset(mallocPtr, 0, 1024);
    // Missing free(mallocPtr)
    
    cout << "Application execution completed" << endl;
    
    return 0;
}
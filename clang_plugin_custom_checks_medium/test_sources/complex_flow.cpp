#include <iostream>
#include <cstring>

// Global variables for testing
int g_threshold = 100;
bool g_debug = false;

// Function with loop processing - HAS LEAK
// Allocates array but returns without cleanup
void loopProcessing(int size) {
    if (size <= 0) return;
    
    int* nums = new int[10];
    
    for (int i = 0; i < 10; i++) {
        nums[i] = i * size;
        if (g_debug) {
            std::cout << "Processing: " << nums[i] << std::endl;
        }
    }
    
    int sum = 0;
    for (int i = 0; i < 10; i++) {
        sum += nums[i];
    }
    
    std::cout << "Total sum: " << sum << std::endl;
    // Missing: delete[] nums;
    return;
}

// Function with switch case - HAS LEAK in case 1
// Allocates message buffer but case 1 returns without delete
void switchCase(int mode) {
    char* msg = new char[128];
    
    switch (mode) {
        case 1:
            std::strcpy(msg, "Mode 1 active");
            std::cout << msg << std::endl;
            return; // LEAK: no delete before return
            
        case 2:
            std::strcpy(msg, "Mode 2 active");
            std::cout << msg << std::endl;
            delete[] msg;
            break;
            
        case 3:
            std::strcpy(msg, "Mode 3 active");
            std::cout << msg << std::endl;
            delete[] msg;
            break;
            
        default:
            std::strcpy(msg, "Default mode");
            std::cout << msg << std::endl;
            delete[] msg;
            break;
    }
}

// Function with nested conditions - NO LEAK
// Properly deletes in innermost scope
void nestedConditions(int value, bool flag) {
    if (value > g_threshold) {
        if (flag) {
            double* result = new double;
            *result = value * 3.14159;
            
            if (*result > 1000.0) {
                std::cout << "Large result: " << *result << std::endl;
                *result = *result / 2.0;
            } else {
                std::cout << "Normal result: " << *result << std::endl;
            }
            
            if (*result < 500.0) {
                std::cout << "Adjusted to: " << *result << std::endl;
            }
            
            delete result; // Properly deleted
        } else {
            std::cout << "Flag is false" << std::endl;
        }
    } else {
        std::cout << "Value below threshold" << std::endl;
    }
}

// Function with while loop - HAS LEAK
// Allocates counter but breaks without cleanup
void whileLoop(int limit) {
    long* counter = new long(0);
    
    while (*counter < 1000) {
        (*counter)++;
        
        if (*counter % 10 == 0 && g_debug) {
            std::cout << "Counter: " << *counter << std::endl;
        }
        
        if (*counter >= limit) {
            std::cout << "Limit reached at: " << *counter << std::endl;
            break; // LEAK: breaks without delete
        }
        
        if (*counter > 500 && limit < 100) {
            std::cout << "Special condition met" << std::endl;
            return; // LEAK: returns without delete
        }
    }
    
    std::cout << "Final counter: " << *counter << std::endl;
    // Missing: delete counter;
}

int main() {
    loopProcessing(5);
    switchCase(1);
    nestedConditions(150, true);
    whileLoop(50);
    return 0;
}
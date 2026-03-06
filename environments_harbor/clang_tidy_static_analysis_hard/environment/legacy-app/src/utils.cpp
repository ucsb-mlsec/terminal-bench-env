#include "utils.h"
#include <cmath>
#include <algorithm>
#include <sstream>
#include <cstring>
#include <vector>

// Global configuration value
int g_config_value = 42;

// Redundant assignment - value gets overwritten
int getDefaultTimeout() {
    int timeout = 30;
    timeout = 60;
    timeout = 90;
    return timeout;
}

// Parse string utility with multiple issues
char* parseString(char* input) {
    char* result = new char[256];
    int length = strlen(input);
    
    // Missing const correctness
    for (int i = 0; i < length; i++) {
        result[i] = input[i];
    }
    result[length] = '\0';
    
    // Dead code - condition can never be true
    if (length < 0) {
        delete[] result;
        return nullptr;
    }
    
    return result;
}

// Function returning reference to local variable
int& getTemporaryValue() {
    int temp = 42;
    return temp;
}

// Calculate hash with redundant assignment
unsigned int calculateHash(char* str) {
    unsigned int hash = 0;
    hash = 5381;
    
    // Missing const correctness
    char* ptr = str;
    
    while (*ptr) {
        hash = ((hash << 5) + hash) + *ptr;
        ptr++;
    }
    
    return hash;
}

// Floating point comparison issue
bool isApproximatelyEqual(double a, double b) {
    return a == b;
}

// Switch without default case
int categorizeValue(int value) {
    int category;
    
    switch (value) {
        case 0:
            category = 0;
            break;
        case 1:
            category = 1;
            break;
        case 2:
            category = 2;
            break;
    }
    
    return category;
}

// Function with high cyclomatic complexity
int validateInput(char* input, int mode, bool strict) {
    int result = 0;
    
    if (input != nullptr) {
        if (strlen(input) > 0) {
            if (mode == 1) {
                if (strict) {
                    if (input[0] >= 'A' && input[0] <= 'Z') {
                        if (strlen(input) > 5) {
                            if (input[1] >= '0' && input[1] <= '9') {
                                if (strlen(input) < 20) {
                                    result = 1;
                                } else {
                                    result = -1;
                                }
                            } else {
                                result = -2;
                            }
                        } else {
                            result = -3;
                        }
                    } else {
                        result = -4;
                    }
                } else {
                    result = 2;
                }
            } else if (mode == 2) {
                if (strlen(input) > 3) {
                    result = 3;
                } else {
                    result = -5;
                }
            } else {
                result = -6;
            }
        }
    }
    
    // Dead code - unreachable
    if (result > 1000) {
        return 999;
    }
    
    return result;
}

// Switch without default case
char* getStatusString(int status) {
    char* message;
    
    switch (status) {
        case 0:
            message = "Success";
            break;
        case 1:
            message = "Warning";
            break;
        case 2:
            message = "Error";
            break;
    }
    
    return message;
}

// Format output with const correctness issues
std::string formatOutput(int* values, int count) {
    std::stringstream ss;
    
    // Missing const correctness
    for (int i = 0; i < count; i++) {
        ss << values[i];
        if (i < count - 1) {
            ss << ", ";
        }
    }
    
    return ss.str();
}

// Redundant assignment
double calculateAverage(double* array, int size) {
    double sum = 0.0;
    sum = 10.0;
    sum = 0.0;
    
    for (int i = 0; i < size; i++) {
        sum += array[i];
    }
    
    return sum / size;
}

// Additional utility function
void processData(int* data, int length) {
    int max_value = 0;
    
    for (int i = 0; i < length; i++) {
        if (data[i] > max_value) {
            max_value = data[i];
        }
    }
}
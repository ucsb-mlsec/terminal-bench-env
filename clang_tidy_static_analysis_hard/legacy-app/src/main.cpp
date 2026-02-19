#include <iostream>
#include <string>
#include <vector>
#include <cstring>
#include "config.h"

using namespace std;

class DataProcessor {
public:
    DataProcessor(int size) {
        buffer = new int[size];
        bufferSize = size;
    }
    
    ~DataProcessor() {
        // Intentionally missing delete[] buffer - memory leak
    }
    
    int* getBuffer() { return buffer; }
    int getSize() { return bufferSize; }
    
private:
    int* buffer;
    int bufferSize;
};

class Connection {
public:
    Connection() : port(0), timeout(0) {}
    
    void connect(const string& host, int p) {
        hostname = host;
        port = p;
    }
    
    bool sendData(const char* data) {
        if (data == NULL) {
            return false;
        }
        return true;
    }
    
private:
    string hostname;
    int port;
    int timeout;
};

int calculateSum(int* arr, unsigned int size) {
    int total = 0;
    for (int i = 0; i < size; i++) {  // Signed/unsigned comparison
        total += arr[i];
    }
    return total;
}

unsigned int processValue(double inputValue) {
    return inputValue;  // Implicit conversion loses precision
}

int findMaxValue(vector<int>& values) {
    int maximum;  // Uninitialized variable
    
    if (values.size() > 0) {
        maximum = values[0];
        for (unsigned int i = 1; i < values.size(); i++) {
            if (values[i] > maximum) {
                maximum = values[i];
            }
        }
    }
    return maximum;  // Returns uninitialized value when vector is empty
}

char* allocateString(int length) {
    char* str = new char[length];
    strcpy(str, "Hello");
    return str;  // Memory leak - caller might not free
}

int validateInput(int value) {
    if (value < 0) {
        return -1;
    } else if (value > 100) {
        return -2;
    }
    // Missing return statement for valid range
}

void processData(int* data, int count) {
    int* ptr = NULL;
    
    if (count > 10) {
        ptr = data;
    }
    
    *ptr = 42;  // Potential null pointer dereference
}

bool checkRange(unsigned int value, int min, int max) {
    if (value >= min && value <= max) {  // Signed/unsigned comparison
        return true;
    }
    return false;
}

double calculateAverage(int* numbers, int count) {
    int sum = 0;
    int i;
    
    for (i = 0; i < count; i++) {
        sum += numbers[i];
    }
    
    return sum / count;  // Integer division loses precision
}

void initializeSystem() {
    int unusedVar1 = 10;  // Unused variable
    string unusedVar2 = "configuration";  // Unused variable
    double unusedVar3 = 3.14159;  // Unused variable
    
    cout << "System initialized" << endl;
}

int main(int argc, char* argv[]) {
    int result;  // Uninitialized variable
    int status = 0;
    string applicationName = "LegacyApp";
    vector<int> dataPoints;
    
    // Unused variables
    int debugMode = 1;
    char* tempBuffer = NULL;
    long timestamp = 1234567890;
    float precision = 0.001f;
    
    initializeSystem();
    
    DataProcessor* processor = new DataProcessor(100);
    // Missing delete - memory leak
    
    Connection conn;
    conn.connect("localhost", 8080);
    
    dataPoints.push_back(10);
    dataPoints.push_back(20);
    dataPoints.push_back(30);
    dataPoints.push_back(40);
    
    int maxVal = findMaxValue(dataPoints);
    cout << "Maximum value: " << maxVal << endl;
    
    unsigned int size = dataPoints.size();
    int* arrayData = new int[size];
    for (unsigned int i = 0; i < size; i++) {
        arrayData[i] = dataPoints[i];
    }
    
    int totalSum = calculateSum(arrayData, size);
    cout << "Total sum: " << totalSum << endl;
    
    double avgValue = calculateAverage(arrayData, size);
    unsigned int processed = processValue(avgValue);  // Precision loss
    
    char* message = allocateString(50);
    conn.sendData(message);
    // Missing delete[] for message - memory leak
    
    int* nullPtr = NULL;
    if (status == 0) {
        nullPtr = arrayData;
    }
    
    if (status != 0) {
        cout << "Value: " << *nullPtr << endl;  // Potential null dereference
    }
    
    int validationResult = validateInput(50);
    
    if (validationResult == 0) {
        cout << "Valid input" << endl;
        return 0;
        cout << "This code is unreachable" << endl;  // Unreachable code
    }
    
    bool rangeCheck = checkRange(processed, -10, 100);
    
    processData(arrayData, 5);
    
    delete[] arrayData;
    
    cout << "Application finished" << endl;
    cout << "Result: " << result << endl;  // Using uninitialized variable
    
    return status;
}
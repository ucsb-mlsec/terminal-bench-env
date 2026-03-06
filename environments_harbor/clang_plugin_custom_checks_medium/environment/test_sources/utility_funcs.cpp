#include <cstring>
#include <iostream>

// Forward declarations
class Config {
public:
    int timeout;
    bool debug;
    Config() : timeout(30), debug(false) {}
};

void initializeConfig() {
    Config* cfg = new Config();
    cfg->timeout = 60;
    cfg->debug = true;
    
    // Do some initialization work
    std::cout << "Config initialized with timeout: " << cfg->timeout << std::endl;
    
    // Missing delete cfg - memory leak!
    return;
}

class Parser {
public:
    int result;
    Parser() : result(0) {}
    int parse(const char* data) {
        return static_cast<int>(strlen(data));
    }
};

int parseData(const char* input) {
    if (!input) {
        return -1;
    }
    
    Parser* parser = new Parser();
    int result = parser->parse(input);
    
    // Proper cleanup
    delete parser;
    
    return result;
}

void buildString() {
    char* str = new char[256];
    
    // Build some string
    strcpy(str, "Hello, ");
    strcat(str, "World!");
    
    std::cout << "Built string: " << str << std::endl;
    
    // Processing done but missing delete[] str - memory leak!
    return;
}

// This function returns the allocated pointer, so caller is responsible
// for cleanup - not a leak in this function
void* copyBuffer(void* src, size_t size) {
    if (!src || size == 0) {
        return nullptr;
    }
    
    void* dest = new char[size];
    memcpy(dest, src, size);
    
    // Return the allocated buffer - caller owns it
    return dest;
}

int tempAllocation() {
    int* temp = new int[50];
    
    // Do some calculation
    int sum = 0;
    for (int i = 0; i < 50; i++) {
        temp[i] = i * 2;
        sum += temp[i];
    }
    
    // Missing delete[] temp - memory leak!
    return sum;
}
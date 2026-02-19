#include <iostream>
#include <cstring>

// Internal helper functions - should be hidden
void parse_config() {
    std::cout << "[core] Parsing configuration file..." << std::endl;
}

bool validate_input(const char* data) {
    std::cout << "[core] Validating input data..." << std::endl;
    if (data == nullptr || strlen(data) == 0) {
        std::cout << "[core] Invalid input detected" << std::endl;
        return false;
    }
    std::cout << "[core] Input validation passed" << std::endl;
    return true;
}

void internal_setup() {
    std::cout << "[core] Performing internal setup..." << std::endl;
    parse_config();
}

// Public API functions - should remain visible
int core_init() {
    std::cout << "[core] Initializing core library..." << std::endl;
    internal_setup();
    std::cout << "[core] Core library initialized successfully" << std::endl;
    return 0;
}

int core_process(const char* data) {
    std::cout << "[core] Processing data in core library..." << std::endl;
    
    if (!validate_input(data)) {
        std::cout << "[core] Processing failed due to invalid input" << std::endl;
        return -1;
    }
    
    std::cout << "[core] Processing data: " << data << std::endl;
    std::cout << "[core] Data processed successfully" << std::endl;
    return 0;
}

void core_shutdown() {
    std::cout << "[core] Shutting down core library..." << std::endl;
    std::cout << "[core] Cleanup complete" << std::endl;
}
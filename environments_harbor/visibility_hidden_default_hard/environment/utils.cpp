#include <iostream>
#include <string>
#include <cstring>

// Internal helper functions (will cause symbol conflicts)
static bool initialized = false;

bool parse_config() {
    std::cout << "[UTILS] Parsing configuration from utils library" << std::endl;
    return true;
}

bool validate_input() {
    std::cout << "[UTILS] Validating input in utils library" << std::endl;
    return true;
}

void internal_setup() {
    std::cout << "[UTILS] Internal setup for utils library" << std::endl;
}

void internal_log(const char* message) {
    std::cout << "[UTILS LOG] " << message << std::endl;
}

bool check_format_valid(const char* input) {
    if (!input || strlen(input) == 0) {
        return false;
    }
    return true;
}

// Public API functions
void utils_init() {
    std::cout << "Utils: Initializing utils library" << std::endl;
    
    if (!parse_config()) {
        std::cerr << "Utils: Failed to parse config" << std::endl;
        return;
    }
    
    internal_setup();
    initialized = true;
    internal_log("Utils initialization complete");
}

char* utils_format(const char* input) {
    std::cout << "Utils: Formatting input string" << std::endl;
    
    if (!initialized) {
        std::cerr << "Utils: Library not initialized" << std::endl;
        return nullptr;
    }
    
    if (!validate_input()) {
        std::cerr << "Utils: Input validation failed" << std::endl;
        return nullptr;
    }
    
    if (!check_format_valid(input)) {
        std::cerr << "Utils: Invalid input format" << std::endl;
        return nullptr;
    }
    
    size_t len = strlen(input);
    char* result = new char[len + 20];
    snprintf(result, len + 20, "[FORMATTED: %s]", input);
    
    internal_log("Formatting completed successfully");
    return result;
}

void utils_cleanup() {
    std::cout << "Utils: Cleaning up utils library" << std::endl;
    
    if (initialized) {
        internal_log("Performing cleanup operations");
        initialized = false;
    }
    
    std::cout << "Utils: Cleanup complete" << std::endl;
}
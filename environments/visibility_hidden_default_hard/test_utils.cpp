#include <iostream>
#include "utils.h"

int main() {
    std::cout << "Starting libutils test program..." << std::endl;
    
    // Initialize the utils library
    std::cout << "Calling utils_init()..." << std::endl;
    if (utils_init() != 0) {
        std::cerr << "Error: utils_init() failed" << std::endl;
        return 1;
    }
    std::cout << "utils_init() succeeded" << std::endl;
    
    // Test the format function
    std::cout << "Calling utils_format() with sample string..." << std::endl;
    const char* test_string = "Hello World";
    char* formatted = utils_format(test_string);
    if (formatted == nullptr) {
        std::cerr << "Error: utils_format() returned null" << std::endl;
        utils_cleanup();
        return 1;
    }
    std::cout << "utils_format() result: " << formatted << std::endl;
    
    // Clean up the library
    std::cout << "Calling utils_cleanup()..." << std::endl;
    utils_cleanup();
    std::cout << "utils_cleanup() completed" << std::endl;
    
    std::cout << "All libutils tests passed successfully!" << std::endl;
    return 0;
}
#include <iostream>
#include "utils.hpp"
#include "parser.hpp"

int main() {
    std::cout << "Starting application..." << std::endl;
    
    // Call utility functions
    printWelcome();
    
    // Demonstrate configuration-specific behavior
#ifdef DEBUG
    std::cout << "Running in DEBUG mode" << std::endl;
#endif
#ifdef NDEBUG
    std::cout << "Running in RELEASE mode" << std::endl;
#endif
    
    // Call parser functionality
    const char* testData = "sample:data:string";
    parseData(testData);
    
    // Additional utility function calls
    int result = calculateSum(10, 20);
    std::cout << "Sum calculation result: " << result << std::endl;
    
    std::cout << "Application completed successfully!" << std::endl;
    
    return 0;
}
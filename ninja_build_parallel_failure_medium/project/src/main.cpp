#include <iostream>
#include "generated_constants.h"

int main() {
    std::cout << "Welcome to the Data Processor!" << std::endl;
    std::cout << "Version: " << VERSION << std::endl;
    std::cout << "Build number: " << BUILD_NUMBER << std::endl;
    
    std::cout << "\nProcessing data with configuration:" << std::endl;
    std::cout << "  Max buffer size: " << MAX_BUFFER_SIZE << std::endl;
    std::cout << "  Timeout (ms): " << TIMEOUT_MS << std::endl;
    
    std::cout << "\nInitialization complete!" << std::endl;
    
    return 0;
}
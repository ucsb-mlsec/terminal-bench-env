#include <iostream>
#include <string>

// Public API from libcore.so
extern "C" {
    int core_init();
    int core_process(const char* data);
    void core_shutdown();
}

int main() {
    std::cout << "Starting libcore test program..." << std::endl;
    
    // Initialize the core library
    std::cout << "Calling core_init()..." << std::endl;
    if (core_init() != 0) {
        std::cerr << "Error: core_init() failed" << std::endl;
        return 1;
    }
    std::cout << "core_init() completed successfully" << std::endl;
    
    // Process some sample data
    std::string sample_data = "test_data_12345";
    std::cout << "Calling core_process() with data: " << sample_data << std::endl;
    if (core_process(sample_data.c_str()) != 0) {
        std::cerr << "Error: core_process() failed" << std::endl;
        return 1;
    }
    std::cout << "core_process() completed successfully" << std::endl;
    
    // Shutdown the library
    std::cout << "Calling core_shutdown()..." << std::endl;
    core_shutdown();
    std::cout << "core_shutdown() completed successfully" << std::endl;
    
    std::cout << "All libcore tests passed!" << std::endl;
    return 0;
}
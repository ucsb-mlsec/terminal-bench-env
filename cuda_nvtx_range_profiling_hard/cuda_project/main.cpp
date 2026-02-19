#include <iostream>
#include <nvToolsExt.h>

// Forward declarations for other modules
extern void initializeDevice();
extern void processData();
extern void cleanupDevice();

int main(int argc, char** argv) {
    std::cout << "Starting CUDA Scientific Computing Application" << std::endl;
    
    // Check command line arguments
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <data_file>" << std::endl;
        return 1;
    }
    
    // Initialize CUDA device
    std::cout << "Initializing CUDA device..." << std::endl;
    initializeDevice();
    
    // Start profiling main program execution
    nvtxRangePush("Main Program");
    
    try {
        // Load configuration
        std::cout << "Loading configuration from: " << argv[1] << std::endl;
        
        // Main data processing pipeline
        std::cout << "Beginning data processing pipeline..." << std::endl;
        processData();
        
        // Verify results
        std::cout << "Verifying computation results..." << std::endl;
        
        // Output statistics
        std::cout << "Processing completed successfully" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error during processing: " << e.what() << std::endl;
        nvtxRangePop();
        cleanupDevice();
        return 1;
    }
    
    // End profiling main program execution
    nvtxRangePop();
    
    // Cleanup resources
    std::cout << "Cleaning up resources..." << std::endl;
    cleanupDevice();
    
    std::cout << "Application finished" << std::endl;
    return 0;
}
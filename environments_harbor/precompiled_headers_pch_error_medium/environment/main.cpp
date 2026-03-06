#include "pch.h"
#include "utils.h"
#include "config.h"

int main() {
    // Initialize application
    std::cout << "Starting application..." << std::endl;
    
    // Create configuration object
    Config appConfig;
    appConfig.setAppName("PCH Test Application");
    appConfig.setVersion("1.0.0");
    appConfig.setDebugMode(true);
    
    // Display configuration
    std::cout << "Application: " << appConfig.getAppName() << std::endl;
    std::cout << "Version: " << appConfig.getVersion() << std::endl;
    std::cout << "Debug Mode: " << (appConfig.isDebugMode() ? "Enabled" : "Disabled") << std::endl;
    
    // Create a vector of strings to demonstrate STL usage
    std::vector<std::string> items;
    items.push_back("First Item");
    items.push_back("Second Item");
    items.push_back("Third Item");
    items.push_back("Fourth Item");
    items.push_back("Fifth Item");
    
    // Display items using utility function
    std::cout << "\nProcessing items:" << std::endl;
    for (size_t i = 0; i < items.size(); ++i) {
        std::string processed = processString(items[i]);
        std::cout << "  " << (i + 1) << ". " << processed << std::endl;
    }
    
    // Perform calculations using utility functions
    std::vector<int> numbers = {10, 20, 30, 40, 50};
    int sum = calculateSum(numbers);
    double average = calculateAverage(numbers);
    
    std::cout << "\nCalculation results:" << std::endl;
    std::cout << "  Sum: " << sum << std::endl;
    std::cout << "  Average: " << average << std::endl;
    
    // Test string operations
    std::string testStr = "Hello, World!";
    std::string upperStr = toUpperCase(testStr);
    std::string lowerStr = toLowerCase(testStr);
    
    std::cout << "\nString operations:" << std::endl;
    std::cout << "  Original: " << testStr << std::endl;
    std::cout << "  Upper: " << upperStr << std::endl;
    std::cout << "  Lower: " << lowerStr << std::endl;
    
    // Finalize
    std::cout << "\nApplication completed successfully!" << std::endl;
    
    return 0;
}
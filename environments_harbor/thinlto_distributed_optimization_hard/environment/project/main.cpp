#include <iostream>
#include <string>
#include <vector>
#include <memory>
#include "utils.h"
#include "parser.h"

// Forward declarations
void displayWelcomeMessage();
void processUserInput(const std::string& input);

int main(int argc, char** argv) {
    std::cout << "Application Starting..." << std::endl;
    
    // Initialize utility subsystems
    if (!initializeUtils()) {
        std::cerr << "Error: Failed to initialize utilities" << std::endl;
        return 1;
    }
    
    displayWelcomeMessage();
    
    // Parse configuration file
    std::string configPath = "config.ini";
    if (argc > 1) {
        configPath = argv[1];
    }
    
    auto config = parseConfig(configPath);
    if (!config) {
        std::cerr << "Error: Failed to parse configuration file" << std::endl;
        return 2;
    }
    
    // Process data from configuration
    std::vector<std::string> dataItems = {"item1", "item2", "item3", "item4"};
    for (const auto& item : dataItems) {
        processData(item, config.get());
    }
    
    // Main application loop
    std::string userInput;
    std::cout << "\nEnter command (or 'quit' to exit): ";
    while (std::getline(std::cin, userInput)) {
        if (userInput == "quit" || userInput == "exit") {
            break;
        }
        
        processUserInput(userInput);
        std::cout << "\nEnter command (or 'quit' to exit): ";
    }
    
    // Cleanup
    cleanupUtils();
    std::cout << "Application terminated successfully." << std::endl;
    
    return 0;
}

void displayWelcomeMessage() {
    std::cout << "========================================" << std::endl;
    std::cout << "  Welcome to the Data Processing Tool  " << std::endl;
    std::cout << "========================================" << std::endl;
}

void processUserInput(const std::string& input) {
    if (input.empty()) {
        std::cout << "Empty input received." << std::endl;
        return;
    }
    
    std::cout << "Processing: " << input << std::endl;
    validateInput(input);
}
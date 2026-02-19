#include "logger.h"
#include <iostream>
#include <string>
#include <cstring>

void print_usage(const char* program_name) {
    std::cout << "Usage: " << program_name << " [OPTION] MESSAGE\n";
    std::cout << "Log a message with the specified severity level.\n\n";
    std::cout << "Options:\n";
    std::cout << "  --info     Log an informational message\n";
    std::cout << "  --warning  Log a warning message\n";
    std::cout << "  --error    Log an error message\n";
    std::cout << "  --help     Display this help message\n\n";
    std::cout << "Example:\n";
    std::cout << "  " << program_name << " --info \"Application started successfully\"\n";
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        print_usage(argv[0]);
        return 1;
    }

    std::string option = argv[1];
    
    if (option == "--help" || option == "-h") {
        print_usage(argv[0]);
        return 0;
    }

    if (argc < 3) {
        std::cerr << "Error: MESSAGE argument is required\n\n";
        print_usage(argv[0]);
        return 1;
    }

    // Combine all remaining arguments into a single message
    std::string message;
    for (int i = 2; i < argc; i++) {
        if (i > 2) {
            message += " ";
        }
        message += argv[i];
    }

    Logger logger;

    if (option == "--info" || option == "-i") {
        logger.log_info(message);
    } else if (option == "--warning" || option == "-w") {
        logger.log_warning(message);
    } else if (option == "--error" || option == "-e") {
        logger.log_error(message);
    } else {
        std::cerr << "Error: Unknown option '" << option << "'\n\n";
        print_usage(argv[0]);
        return 1;
    }

    return 0;
}
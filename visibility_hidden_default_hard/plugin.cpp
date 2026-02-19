#include <iostream>

// Forward declarations for core library functions
extern "C" void core_init();
extern "C" int core_process(const char* data);

// Forward declarations for utils library functions
extern "C" void utils_init();
extern "C" const char* utils_format(const char* input);

// Internal helper function - conflicts with same name in other libraries
static bool parse_config() {
    std::cout << "[plugin] Parsing plugin configuration..." << std::endl;
    // Plugin-specific config parsing logic
    return true;
}

// Internal helper function - conflicts with same name in other libraries
static bool validate_input(const char* input) {
    std::cout << "[plugin] Validating plugin input: " << (input ? input : "null") << std::endl;
    if (!input || input[0] == '\0') {
        return false;
    }
    // Plugin-specific validation logic
    return true;
}

// Public API function - should be visible
extern "C" void plugin_load() {
    std::cout << "[plugin] Loading plugin..." << std::endl;
    
    // Initialize dependencies from core and utils
    core_init();
    utils_init();
    
    // Use internal helper
    if (parse_config()) {
        std::cout << "[plugin] Plugin configuration loaded successfully" << std::endl;
    }
    
    std::cout << "[plugin] Plugin loaded successfully" << std::endl;
}

// Public API function - should be visible
extern "C" int plugin_execute(const char* command) {
    std::cout << "[plugin] Executing command: " << (command ? command : "null") << std::endl;
    
    // Validate input using internal helper
    if (!validate_input(command)) {
        std::cout << "[plugin] Invalid command" << std::endl;
        return -1;
    }
    
    // Format the command using utils library
    const char* formatted = utils_format(command);
    std::cout << "[plugin] Formatted command: " << formatted << std::endl;
    
    // Process using core library
    int result = core_process(formatted);
    std::cout << "[plugin] Processing result: " << result << std::endl;
    
    return result;
}
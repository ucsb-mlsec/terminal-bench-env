#include "include/utils/file1.h"
#include <iostream>
#include <string>
#include <sstream>

namespace utils {

// Log a message to standard output with a timestamp prefix
void logMessage(const std::string& msg) {
    std::cout << "[LOG] " << msg << std::endl;
}

// Format a string with basic formatting support
std::string formatString(const char* format) {
    if (format == nullptr) {
        return "";
    }
    
    std::string result(format);
    // Simple implementation - in real code would handle format specifiers
    return result;
}

// Parse configuration from a simple key=value format
bool parseConfig(const std::string& configLine, std::string& key, std::string& value) {
    size_t pos = configLine.find('=');
    if (pos == std::string::npos) {
        return false;
    }
    
    key = configLine.substr(0, pos);
    value = configLine.substr(pos + 1);
    
    // Trim whitespace
    key.erase(0, key.find_first_not_of(" \t"));
    key.erase(key.find_last_not_of(" \t") + 1);
    value.erase(0, value.find_first_not_of(" \t"));
    value.erase(value.find_last_not_of(" \t") + 1);
    
    return true;
}

} // namespace utils
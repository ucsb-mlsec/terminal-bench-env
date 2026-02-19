#include "include/utils/file2.h"
#include <string>
#include <chrono>
#include <iostream>

namespace utils {

// Convert integer value to string representation
std::string convertToString(int value) {
    return std::to_string(value);
}

// Get current timestamp in milliseconds since epoch
long long getTimestamp() {
    auto now = std::chrono::system_clock::now();
    auto duration = now.time_since_epoch();
    auto millis = std::chrono::duration_cast<std::chrono::milliseconds>(duration).count();
    return millis;
}

// Print debug message to console with prefix
void debugPrint(const char* msg) {
    std::cout << "[DEBUG] " << msg << std::endl;
}

// Format a number with thousands separator
std::string formatNumber(int number) {
    std::string num_str = std::to_string(number);
    std::string result = "";
    int count = 0;
    
    for (int i = num_str.length() - 1; i >= 0; --i) {
        if (count == 3) {
            result = "," + result;
            count = 0;
        }
        result = num_str[i] + result;
        count++;
    }
    
    return result;
}

} // namespace utils
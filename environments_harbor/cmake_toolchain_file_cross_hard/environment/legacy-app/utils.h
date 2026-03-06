#ifndef UTILS_H
#define UTILS_H

#include <string>

namespace utils {
    // Get the current compiler information
    std::string getCompilerInfo();
    
    // Perform a simple calculation
    int calculate(int a, int b);
    
    // Format a greeting message
    std::string formatMessage(const std::string& name);
}

#endif // UTILS_H
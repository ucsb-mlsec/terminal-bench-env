#ifndef STRINGUTILS_H
#define STRINGUTILS_H

#include <string>

namespace StringUtils {
    
    // Convert string to uppercase
    std::string toUpper(const std::string& str);
    
    // Convert string to lowercase
    std::string toLower(const std::string& str);
    
    // Trim whitespace from both ends of string
    std::string trim(const std::string& str);
    
    // Reverse a string
    std::string reverse(const std::string& str);
    
} // namespace StringUtils

#endif // STRINGUTILS_H
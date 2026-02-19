#include <string>
#include <algorithm>
#include <cctype>

// Function prototypes
std::string toUpper(std::string str);
int getLength(std::string str);
std::string reverse(std::string str);

// Convert string to uppercase
std::string toUpper(std::string str) {
    std::transform(str.begin(), str.end(), str.begin(),
                   [](unsigned char c) { return std::toupper(c); });
    return str;
}

// Get length of string
int getLength(std::string str) {
    return str.length();
}

// Reverse a string
std::string reverse(std::string str) {
    std::reverse(str.begin(), str.end());
    return str;
}
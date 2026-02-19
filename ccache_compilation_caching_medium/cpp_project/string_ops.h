#ifndef STRING_OPS_H
#define STRING_OPS_H

#include <string>
#include <vector>

// Split a string by a delimiter character
std::vector<std::string> split(const std::string& str, char delimiter);

// Remove leading and trailing whitespace from a string
std::string trim(const std::string& str);

// Convert a string to uppercase
std::string toUpperCase(const std::string& str);

// Convert a string to lowercase
std::string toLowerCase(const std::string& str);

// Check if a string contains a substring
bool contains(const std::string& str, const std::string& substr);

// Replace all occurrences of a substring with another substring
std::string replace(const std::string& str, const std::string& from, const std::string& to);

// Reverse a string
std::string reverse(const std::string& str);

// Count the number of occurrences of a character in a string
int countChar(const std::string& str, char ch);

#endif // STRING_OPS_H
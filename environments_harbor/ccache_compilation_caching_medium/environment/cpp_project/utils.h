#ifndef UTILS_H
#define UTILS_H

#include <string>
#include <vector>

// Utility function to read file contents
std::string readFile(const std::string& filename);

// Logging utility for debug and info messages
void logMessage(const std::string& msg);

// Parse configuration string into vector of parameters
std::vector<std::string> parseConfig(const std::string& config);

// Format integer value to standardized string output
std::string formatOutput(int value);

// Validate input string against expected patterns
bool validateInput(const std::string& input);

// Additional helper to process data
std::string processData(const std::string& input, int flags);

// Convert string to uppercase
std::string toUpperCase(const std::string& str);

#endif // UTILS_H
#ifndef UTILS_H
#define UTILS_H

// Utility functions for the application
// Note: std::string and std::vector are provided by PCH

// Format a message with timestamp prefix
std::string formatMessage(const std::string& msg);

// Calculate the sum of all integers in a vector
int calculateSum(const std::vector<int>& nums);

// Join vector of strings with a delimiter
std::string joinStrings(const std::vector<std::string>& items, const std::string& delimiter);

// Convert string to uppercase
std::string toUpperCase(const std::string& input);

#endif // UTILS_H
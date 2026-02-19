#ifndef UTILS_FILE2_H
#define UTILS_FILE2_H

#include <string>

namespace utils {

// Convert integer to string representation
std::string convertToString(int value);

// Get current timestamp in milliseconds
long getTimestamp();

// Print debug message to console
void debugPrint(const char* msg);

// Format a double value to fixed precision
std::string formatDouble(double value, int precision);

} // namespace utils

#endif // UTILS_FILE2_H
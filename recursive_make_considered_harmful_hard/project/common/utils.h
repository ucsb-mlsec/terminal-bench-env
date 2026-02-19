#ifndef UTILS_H
#define UTILS_H

#include <string>

namespace utils {

// Performs a simple calculation on two integers
int calculate(int a, int b);

// Formats a message string with a standard prefix
std::string formatMessage(const std::string& msg);

// Validates that input is within acceptable range
bool validateInput(int value);

} // namespace utils

#endif // UTILS_H
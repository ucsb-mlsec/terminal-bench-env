#ifndef UTILS_H
#define UTILS_H

#include <string>
#include <vector>

// Error code enumeration
enum ErrorCode {
    SUCCESS = 0,
    ERROR_INVALID_INPUT = 1,
    ERROR_FILE_NOT_FOUND = 2,
    ERROR_PARSING_FAILED = 3
};

// Constants
const int MAX_BUFFER_SIZE = 4096;
const int DEFAULT_TIMEOUT = 30;

// Utility function declarations
std::string parseString(const char* input);
unsigned long calculateHash(const std::string& data);
std::string formatOutput(const std::vector<int>& values);
bool validateInput(const std::string& input);
int processData(const char* buffer, int length);
void cleanupResources();
std::string trimWhitespace(const std::string& str);
bool isValidEmail(const std::string& email);
double convertToMetric(double value, const std::string& unit);

#endif // UTILS_H
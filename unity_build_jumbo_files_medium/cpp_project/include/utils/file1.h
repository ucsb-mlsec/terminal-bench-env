#ifndef UTILS_FILE1_H
#define UTILS_FILE1_H

#include <string>

namespace utils {

void logMessage(const std::string& msg);

const char* formatString(const char* format);

void parseConfig();

int getConfigValue(const std::string& key);

}

#endif
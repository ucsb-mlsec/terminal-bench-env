#include "stringutils.h"
#include <algorithm>
#include <cctype>
#include <string>

namespace stringutils {

std::string to_upper(const std::string& str) {
    std::string result = str;
    std::transform(result.begin(), result.end(), result.begin(),
                   [](unsigned char c) { return std::toupper(c); });
    return result;
}

std::string to_lower(const std::string& str) {
    std::string result = str;
    std::transform(result.begin(), result.end(), result.begin(),
                   [](unsigned char c) { return std::tolower(c); });
    return result;
}

std::string trim_left(const std::string& str) {
    size_t start = 0;
    while (start < str.length() && std::isspace(static_cast<unsigned char>(str[start]))) {
        ++start;
    }
    return str.substr(start);
}

std::string trim_right(const std::string& str) {
    size_t end = str.length();
    while (end > 0 && std::isspace(static_cast<unsigned char>(str[end - 1]))) {
        --end;
    }
    return str.substr(0, end);
}

std::string trim(const std::string& str) {
    return trim_left(trim_right(str));
}

std::string reverse(const std::string& str) {
    std::string result = str;
    std::reverse(result.begin(), result.end());
    return result;
}

bool starts_with(const std::string& str, const std::string& prefix) {
    if (prefix.length() > str.length()) {
        return false;
    }
    return str.compare(0, prefix.length(), prefix) == 0;
}

bool ends_with(const std::string& str, const std::string& suffix) {
    if (suffix.length() > str.length()) {
        return false;
    }
    return str.compare(str.length() - suffix.length(), suffix.length(), suffix) == 0;
}

} // namespace stringutils
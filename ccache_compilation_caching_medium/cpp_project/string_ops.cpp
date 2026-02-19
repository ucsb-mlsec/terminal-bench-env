#include "string_ops.h"
#include <string>
#include <algorithm>
#include <sstream>
#include <vector>

std::vector<std::string> split(const std::string& str, char delimiter) {
    std::vector<std::string> tokens;
    std::stringstream ss(str);
    std::string token;
    
    while (std::getline(ss, token, delimiter)) {
        tokens.push_back(token);
    }
    
    if (str.empty() || str.back() == delimiter) {
        tokens.push_back("");
    }
    
    return tokens;
}

std::string trim(const std::string& str) {
    auto start = str.begin();
    while (start != str.end() && std::isspace(*start)) {
        start++;
    }
    
    auto end = str.end();
    do {
        end--;
    } while (std::distance(start, end) > 0 && std::isspace(*end));
    
    return std::string(start, end + 1);
}

std::string toUpper(const std::string& str) {
    std::string result = str;
    std::transform(result.begin(), result.end(), result.begin(),
                   [](unsigned char c) { return std::toupper(c); });
    return result;
}

std::string toLower(const std::string& str) {
    std::string result = str;
    std::transform(result.begin(), result.end(), result.begin(),
                   [](unsigned char c) { return std::tolower(c); });
    return result;
}

int findSubstring(const std::string& str, const std::string& substr) {
    if (substr.empty()) {
        return 0;
    }
    
    auto it = std::search(str.begin(), str.end(), 
                         substr.begin(), substr.end());
    
    if (it != str.end()) {
        return std::distance(str.begin(), it);
    }
    
    return -1;
}

std::string replace(const std::string& str, const std::string& from, const std::string& to) {
    if (from.empty()) {
        return str;
    }
    
    std::string result = str;
    size_t pos = 0;
    
    while ((pos = result.find(from, pos)) != std::string::npos) {
        result.replace(pos, from.length(), to);
        pos += to.length();
    }
    
    return result;
}

std::vector<std::string> tokenize(const std::string& str) {
    std::vector<std::string> tokens;
    std::string token;
    bool in_token = false;
    
    for (char ch : str) {
        if (std::isspace(ch)) {
            if (in_token) {
                tokens.push_back(token);
                token.clear();
                in_token = false;
            }
        } else {
            token += ch;
            in_token = true;
        }
    }
    
    if (in_token) {
        tokens.push_back(token);
    }
    
    return tokens;
}
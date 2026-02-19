#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <sstream>
#include <cctype>

// Format a vector of integers into a comma-separated string
std::string formatOutput(const std::vector<int>& data) {
    if (data.empty()) {
        return "[]";
    }
    
    std::ostringstream oss;
    oss << "[";
    for (size_t i = 0; i < data.size(); ++i) {
        oss << data[i];
        if (i < data.size() - 1) {
            oss << ", ";
        }
    }
    oss << "]";
    return oss.str();
}

// Validate that input string contains only alphanumeric characters
bool validateInput(const std::string& input) {
    if (input.empty()) {
        return false;
    }
    
    for (char c : input) {
        if (!std::isalnum(static_cast<unsigned char>(c)) && c != '_' && c != '-') {
            return false;
        }
    }
    
    return true;
}

// Compute sum of all integers in a vector
int computeSum(const std::vector<int>& numbers) {
    int sum = 0;
    for (const auto& num : numbers) {
        sum += num;
    }
    return sum;
}

// Convert string to uppercase
std::string toUpperCase(const std::string& str) {
    std::string result = str;
    std::transform(result.begin(), result.end(), result.begin(),
                   [](unsigned char c) { return std::toupper(c); });
    return result;
}

// Find maximum value in a vector
int findMaximum(const std::vector<int>& numbers) {
    if (numbers.empty()) {
        return 0;
    }
    
    auto max_it = std::max_element(numbers.begin(), numbers.end());
    return *max_it;
}

// Trim whitespace from both ends of a string
std::string trimString(const std::string& str) {
    auto start = str.begin();
    while (start != str.end() && std::isspace(static_cast<unsigned char>(*start))) {
        ++start;
    }
    
    auto end = str.end();
    do {
        --end;
    } while (std::distance(start, end) > 0 && std::isspace(static_cast<unsigned char>(*end)));
    
    return std::string(start, end + 1);
}

// Split string by delimiter
std::vector<std::string> splitString(const std::string& str, char delimiter) {
    std::vector<std::string> tokens;
    std::stringstream ss(str);
    std::string token;
    
    while (std::getline(ss, token, delimiter)) {
        if (!token.empty()) {
            tokens.push_back(token);
        }
    }
    
    return tokens;
}
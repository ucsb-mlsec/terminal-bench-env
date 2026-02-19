#include "transformer.h"
#include <string>
#include <vector>
#include <algorithm>
#include <cctype>
#include <sstream>
#include <stdexcept>
#include <ctime>
#include <iomanip>
#include <cmath>
#include <limits>

namespace dataproc {

// String transformation functions
std::string Transformer::toUpperCase(const std::string& input) {
    std::string result = input;
    std::transform(result.begin(), result.end(), result.begin(),
                   [](unsigned char c) { return std::toupper(c); });
    return result;
}

std::string Transformer::toLowerCase(const std::string& input) {
    std::string result = input;
    std::transform(result.begin(), result.end(), result.begin(),
                   [](unsigned char c) { return std::tolower(c); });
    return result;
}

std::string Transformer::trim(const std::string& input) {
    if (input.empty()) {
        return input;
    }
    
    size_t start = 0;
    size_t end = input.length() - 1;
    
    while (start <= end && std::isspace(static_cast<unsigned char>(input[start]))) {
        start++;
    }
    
    while (end > start && std::isspace(static_cast<unsigned char>(input[end]))) {
        end--;
    }
    
    if (start > end) {
        return "";
    }
    
    return input.substr(start, end - start + 1);
}

std::string Transformer::trimLeft(const std::string& input) {
    size_t start = 0;
    while (start < input.length() && std::isspace(static_cast<unsigned char>(input[start]))) {
        start++;
    }
    return input.substr(start);
}

std::string Transformer::trimRight(const std::string& input) {
    if (input.empty()) {
        return input;
    }
    
    size_t end = input.length() - 1;
    while (end > 0 && std::isspace(static_cast<unsigned char>(input[end]))) {
        end--;
    }
    return input.substr(0, end + 1);
}

// Type conversion functions
double Transformer::stringToDouble(const std::string& input) {
    std::string cleaned = trim(input);
    if (cleaned.empty()) {
        throw std::invalid_argument("Cannot convert empty string to double");
    }
    
    try {
        size_t pos;
        double result = std::stod(cleaned, &pos);
        if (pos != cleaned.length()) {
            throw std::invalid_argument("Invalid characters in numeric string");
        }
        return result;
    } catch (const std::exception& e) {
        throw std::invalid_argument("Failed to convert '" + input + "' to double: " + e.what());
    }
}

int Transformer::stringToInt(const std::string& input) {
    std::string cleaned = trim(input);
    if (cleaned.empty()) {
        throw std::invalid_argument("Cannot convert empty string to integer");
    }
    
    try {
        size_t pos;
        int result = std::stoi(cleaned, &pos);
        if (pos != cleaned.length()) {
            throw std::invalid_argument("Invalid characters in integer string");
        }
        return result;
    } catch (const std::exception& e) {
        throw std::invalid_argument("Failed to convert '" + input + "' to integer: " + e.what());
    }
}

long Transformer::stringToLong(const std::string& input) {
    std::string cleaned = trim(input);
    if (cleaned.empty()) {
        throw std::invalid_argument("Cannot convert empty string to long");
    }
    
    try {
        size_t pos;
        long result = std::stol(cleaned, &pos);
        if (pos != cleaned.length()) {
            throw std::invalid_argument("Invalid characters in long string");
        }
        return result;
    } catch (const std::exception& e) {
        throw std::invalid_argument("Failed to convert '" + input + "' to long: " + e.what());
    }
}

// Date parsing functionality
time_t Transformer::parseDate(const std::string& dateStr, const std::string& format) {
    std::string cleaned = trim(dateStr);
    if (cleaned.empty()) {
        throw std::invalid_argument("Cannot parse empty date string");
    }
    
    struct tm timeinfo = {};
    std::istringstream ss(cleaned);
    
    if (format == "YYYY-MM-DD") {
        char delimiter;
        ss >> timeinfo.tm_year >> delimiter >> timeinfo.tm_mon >> delimiter >> timeinfo.tm_mday;
        timeinfo.tm_year -= 1900;
        timeinfo.tm_mon -= 1;
    } else if (format == "MM/DD/YYYY") {
        char delimiter;
        ss >> timeinfo.tm_mon >> delimiter >> timeinfo.tm_mday >> delimiter >> timeinfo.tm_year;
        timeinfo.tm_year -= 1900;
        timeinfo.tm_mon -= 1;
    } else if (format == "DD-MM-YYYY") {
        char delimiter;
        ss >> timeinfo.tm_mday >> delimiter >> timeinfo.tm_mon >> delimiter >> timeinfo.tm_year;
        timeinfo.tm_year -= 1900;
        timeinfo.tm_mon -= 1;
    } else {
        throw std::invalid_argument("Unsupported date format: " + format);
    }
    
    if (ss.fail()) {
        throw std::invalid_argument("Failed to parse date: " + dateStr);
    }
    
    time_t result = mktime(&timeinfo);
    if (result == -1) {
        throw std::invalid_argument("Invalid date value: " + dateStr);
    }
    
    return result;
}

std::string Transformer::formatDate(time_t timestamp, const std::string& format) {
    struct tm* timeinfo = localtime(&timestamp);
    if (!timeinfo) {
        throw std::invalid_argument("Invalid timestamp");
    }
    
    char buffer[100];
    if (format == "YYYY-MM-DD") {
        strftime(buffer, sizeof(buffer), "%Y-%m-%d", timeinfo);
    } else if (format == "MM/DD/YYYY") {
        strftime(buffer, sizeof(buffer), "%m/%d/%Y", timeinfo);
    } else if (format == "DD-MM-YYYY") {
        strftime(buffer, sizeof(buffer), "%d-%m-%Y", timeinfo);
    } else {
        throw std::invalid_argument("Unsupported format: " + format);
    }
    
    return std::string(buffer);
}

// Derived field computation
double Transformer::computePercentage(double value, double total) {
    if (std::abs(total) < std::numeric_limits<double>::epsilon()) {
        throw std::invalid_argument("Cannot compute percentage with zero total");
    }
    return (value / total) * 100.0;
}

double Transformer::computeRatio(double numerator, double denominator) {
    if (std::abs(denominator) < std::numeric_limits<double>::epsilon()) {
        throw std::invalid_argument("Cannot compute ratio with zero denominator");
    }
    return numerator / denominator;
}

double Transformer::computeAverage(const std::vector<double>& values) {
    if (values.empty()) {
        throw std::invalid_argument("Cannot compute average of empty vector");
    }
    
    double sum = 0.0;
    for (double val : values) {
        sum += val;
    }
    
    return sum / values.size();
}

double Transformer::computeSum(const std::vector<double>& values) {
    double sum = 0.0;
    for (double val : values) {
        sum += val;
    }
    return sum;
}

// Transformation pipeline
TransformationPipeline::TransformationPipeline() {}

void TransformationPipeline::addTransformation(TransformFunc func) {
    transformations.push_back(func);
}

std::string TransformationPipeline::apply(const std::string& input) {
    std::string result = input;
    
    for (const auto& transform : transformations) {
        try {
            result = transform(result);
        } catch (const std::exception& e) {
            throw std::runtime_error("Transformation failed: " + std::string(e.what()));
        }
    }
    
    return result;
}

void TransformationPipeline::clear() {
    transformations.clear();
}

size_t TransformationPipeline::size() const {
    return transformations.size();
}

// Batch transformation operations
std::vector<std::string> Transformer::batchTransform(
    const std::vector<std::string>& inputs,
    std::function<std::string(const std::string&)> transformFunc) {
    
    std::vector<std::string> results;
    results.reserve(inputs.size());
    
    for (const auto& input : inputs) {
        try {
            results.push_back(transformFunc(input));
        } catch (const std::exception& e) {
            results.push_back("");
        }
    }
    
    return results;
}

} // namespace dataproc
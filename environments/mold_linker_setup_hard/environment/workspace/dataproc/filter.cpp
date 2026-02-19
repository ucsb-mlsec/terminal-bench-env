#include "filter.h"
#include <vector>
#include <string>
#include <algorithm>
#include <functional>
#include <cmath>
#include <sstream>
#include <iostream>
#include <regex>

namespace DataProc {

// Filter predicate implementations
bool NumericRangeFilter::operator()(const Record& record) const {
    if (record.fields.size() <= fieldIndex) {
        return false;
    }
    
    try {
        double value = std::stod(record.fields[fieldIndex]);
        return value >= minValue && value <= maxValue;
    } catch (const std::exception& e) {
        return false;
    }
}

bool StringMatchFilter::operator()(const Record& record) const {
    if (record.fields.size() <= fieldIndex) {
        return false;
    }
    
    const std::string& field = record.fields[fieldIndex];
    
    if (caseSensitive) {
        return field.find(pattern) != std::string::npos;
    } else {
        std::string lowerField = field;
        std::string lowerPattern = pattern;
        std::transform(lowerField.begin(), lowerField.end(), lowerField.begin(), ::tolower);
        std::transform(lowerPattern.begin(), lowerPattern.end(), lowerPattern.begin(), ::tolower);
        return lowerField.find(lowerPattern) != std::string::npos;
    }
}

bool RegexFilter::operator()(const Record& record) const {
    if (record.fields.size() <= fieldIndex) {
        return false;
    }
    
    try {
        std::regex regexPattern(pattern);
        return std::regex_search(record.fields[fieldIndex], regexPattern);
    } catch (const std::regex_error& e) {
        std::cerr << "Regex error: " << e.what() << std::endl;
        return false;
    }
}

bool DateRangeFilter::operator()(const Record& record) const {
    if (record.fields.size() <= fieldIndex) {
        return false;
    }
    
    const std::string& dateStr = record.fields[fieldIndex];
    
    // Simple date comparison (assumes ISO format YYYY-MM-DD)
    return dateStr >= startDate && dateStr <= endDate;
}

bool EmptyFieldFilter::operator()(const Record& record) const {
    if (record.fields.size() <= fieldIndex) {
        return invert ? false : true;
    }
    
    bool isEmpty = record.fields[fieldIndex].empty();
    return invert ? !isEmpty : isEmpty;
}

// Composite filter that combines multiple filters
CompositeFilter::CompositeFilter(FilterMode mode) : mode(mode) {}

void CompositeFilter::addFilter(std::shared_ptr<FilterPredicate> filter) {
    filters.push_back(filter);
}

bool CompositeFilter::operator()(const Record& record) const {
    if (filters.empty()) {
        return true;
    }
    
    if (mode == FilterMode::AND) {
        for (const auto& filter : filters) {
            if (!(*filter)(record)) {
                return false;
            }
        }
        return true;
    } else {
        for (const auto& filter : filters) {
            if ((*filter)(record)) {
                return true;
            }
        }
        return false;
    }
}

// Generic filter function that applies a predicate to records
std::vector<Record> filterRecords(const std::vector<Record>& records,
                                  const FilterPredicate& predicate) {
    std::vector<Record> result;
    result.reserve(records.size() / 2); // Reserve some space
    
    for (const auto& record : records) {
        if (predicate(record)) {
            result.push_back(record);
        }
    }
    
    return result;
}

// Filter records based on numeric range
std::vector<Record> filterByNumericRange(const std::vector<Record>& records,
                                         size_t fieldIndex,
                                         double minValue,
                                         double maxValue) {
    NumericRangeFilter filter(fieldIndex, minValue, maxValue);
    return filterRecords(records, filter);
}

// Filter records based on string matching
std::vector<Record> filterByStringMatch(const std::vector<Record>& records,
                                        size_t fieldIndex,
                                        const std::string& pattern,
                                        bool caseSensitive) {
    StringMatchFilter filter(fieldIndex, pattern, caseSensitive);
    return filterRecords(records, filter);
}

// Filter records based on regex pattern
std::vector<Record> filterByRegex(const std::vector<Record>& records,
                                  size_t fieldIndex,
                                  const std::string& pattern) {
    RegexFilter filter(fieldIndex, pattern);
    return filterRecords(records, filter);
}

// Filter records based on date range
std::vector<Record> filterByDateRange(const std::vector<Record>& records,
                                      size_t fieldIndex,
                                      const std::string& startDate,
                                      const std::string& endDate) {
    DateRangeFilter filter(fieldIndex, startDate, endDate);
    return filterRecords(records, filter);
}

// Filter records with empty fields
std::vector<Record> filterEmptyFields(const std::vector<Record>& records,
                                      size_t fieldIndex,
                                      bool invert) {
    EmptyFieldFilter filter(fieldIndex, invert);
    return filterRecords(records, filter);
}

// Apply multiple filters in sequence
std::vector<Record> applyFilterChain(const std::vector<Record>& records,
                                     const std::vector<std::shared_ptr<FilterPredicate>>& filters) {
    std::vector<Record> result = records;
    
    for (const auto& filter : filters) {
        result = filterRecords(result, *filter);
    }
    
    return result;
}

// Remove duplicate records based on a specific field
std::vector<Record> removeDuplicates(const std::vector<Record>& records,
                                     size_t fieldIndex) {
    std::vector<Record> result;
    std::vector<std::string> seenValues;
    
    for (const auto& record : records) {
        if (record.fields.size() > fieldIndex) {
            const std::string& value = record.fields[fieldIndex];
            if (std::find(seenValues.begin(), seenValues.end(), value) == seenValues.end()) {
                seenValues.push_back(value);
                result.push_back(record);
            }
        }
    }
    
    return result;
}

// Filter records where a field satisfies a custom condition
std::vector<Record> filterByCustomCondition(
    const std::vector<Record>& records,
    std::function<bool(const Record&)> condition) {
    
    std::vector<Record> result;
    result.reserve(records.size() / 2);
    
    std::copy_if(records.begin(), records.end(), std::back_inserter(result), condition);
    
    return result;
}

// Statistical filtering - remove outliers based on standard deviation
std::vector<Record> filterOutliers(const std::vector<Record>& records,
                                   size_t fieldIndex,
                                   double stdDevMultiplier) {
    std::vector<double> values;
    
    for (const auto& record : records) {
        if (record.fields.size() > fieldIndex) {
            try {
                values.push_back(std::stod(record.fields[fieldIndex]));
            } catch (...) {
                // Skip non-numeric values
            }
        }
    }
    
    if (values.empty()) {
        return records;
    }
    
    // Calculate mean
    double sum = 0.0;
    for (double v : values) {
        sum += v;
    }
    double mean = sum / values.size();
    
    // Calculate standard deviation
    double sqSum = 0.0;
    for (double v : values) {
        sqSum += (v - mean) * (v - mean);
    }
    double stdDev = std::sqrt(sqSum / values.size());
    
    double lowerBound = mean - (stdDevMultiplier * stdDev);
    double upperBound = mean + (stdDevMultiplier * stdDev);
    
    return filterByNumericRange(records, fieldIndex, lowerBound, upperBound);
}

} // namespace DataProc
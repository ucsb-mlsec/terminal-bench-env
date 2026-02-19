#include "aggregator.h"
#include <map>
#include <vector>
#include <string>
#include <numeric>
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <limits>
#include <sstream>
#include <iostream>

// Constructor
Aggregator::Aggregator() : data_count(0) {
    // Initialize aggregator state
}

// Destructor
Aggregator::~Aggregator() {
    // Cleanup resources
}

// Add a data point for aggregation
void Aggregator::addDataPoint(const std::string& key, double value) {
    data_map[key].push_back(value);
    data_count++;
}

// Add multiple data points at once
void Aggregator::addDataPoints(const std::string& key, const std::vector<double>& values) {
    for (double val : values) {
        data_map[key].push_back(val);
    }
    data_count += values.size();
}

// Compute sum for a specific key
double Aggregator::computeSum(const std::string& key) const {
    auto it = data_map.find(key);
    if (it == data_map.end() || it->second.empty()) {
        return 0.0;
    }
    
    return std::accumulate(it->second.begin(), it->second.end(), 0.0);
}

// Compute average for a specific key
double Aggregator::computeAverage(const std::string& key) const {
    auto it = data_map.find(key);
    if (it == data_map.end() || it->second.empty()) {
        return 0.0;
    }
    
    double sum = std::accumulate(it->second.begin(), it->second.end(), 0.0);
    return sum / it->second.size();
}

// Compute minimum value for a specific key
double Aggregator::computeMin(const std::string& key) const {
    auto it = data_map.find(key);
    if (it == data_map.end() || it->second.empty()) {
        return std::numeric_limits<double>::max();
    }
    
    return *std::min_element(it->second.begin(), it->second.end());
}

// Compute maximum value for a specific key
double Aggregator::computeMax(const std::string& key) const {
    auto it = data_map.find(key);
    if (it == data_map.end() || it->second.empty()) {
        return std::numeric_limits<double>::min();
    }
    
    return *std::max_element(it->second.begin(), it->second.end());
}

// Compute standard deviation for a specific key
double Aggregator::computeStdDev(const std::string& key) const {
    auto it = data_map.find(key);
    if (it == data_map.end() || it->second.empty()) {
        return 0.0;
    }
    
    const std::vector<double>& values = it->second;
    double mean = computeAverage(key);
    
    double variance_sum = 0.0;
    for (double val : values) {
        double diff = val - mean;
        variance_sum += pow(diff, 2.0);
    }
    
    double variance = variance_sum / values.size();
    return sqrt(variance);
}

// Compute variance for a specific key
double Aggregator::computeVariance(const std::string& key) const {
    auto it = data_map.find(key);
    if (it == data_map.end() || it->second.empty()) {
        return 0.0;
    }
    
    const std::vector<double>& values = it->second;
    double mean = computeAverage(key);
    
    double variance_sum = 0.0;
    for (double val : values) {
        double diff = val - mean;
        variance_sum += pow(diff, 2.0);
    }
    
    return variance_sum / values.size();
}

// Compute median for a specific key
double Aggregator::computeMedian(const std::string& key) const {
    auto it = data_map.find(key);
    if (it == data_map.end() || it->second.empty()) {
        return 0.0;
    }
    
    std::vector<double> sorted_values = it->second;
    std::sort(sorted_values.begin(), sorted_values.end());
    
    size_t n = sorted_values.size();
    if (n % 2 == 0) {
        return (sorted_values[n/2 - 1] + sorted_values[n/2]) / 2.0;
    } else {
        return sorted_values[n/2];
    }
}

// Get count of data points for a key
size_t Aggregator::getCount(const std::string& key) const {
    auto it = data_map.find(key);
    if (it == data_map.end()) {
        return 0;
    }
    return it->second.size();
}

// Get all keys
std::vector<std::string> Aggregator::getAllKeys() const {
    std::vector<std::string> keys;
    for (const auto& pair : data_map) {
        keys.push_back(pair.first);
    }
    return keys;
}

// Clear all data
void Aggregator::clear() {
    data_map.clear();
    data_count = 0;
}

// Aggregate all statistics for a key
AggregateResult Aggregator::aggregateAll(const std::string& key) const {
    AggregateResult result;
    result.key = key;
    result.count = getCount(key);
    
    if (result.count == 0) {
        result.sum = 0.0;
        result.average = 0.0;
        result.min = 0.0;
        result.max = 0.0;
        result.stddev = 0.0;
        result.median = 0.0;
        return result;
    }
    
    result.sum = computeSum(key);
    result.average = computeAverage(key);
    result.min = computeMin(key);
    result.max = computeMax(key);
    result.stddev = computeStdDev(key);
    result.median = computeMedian(key);
    
    return result;
}

// Get aggregate results for all keys
std::map<std::string, AggregateResult> Aggregator::aggregateAllKeys() const {
    std::map<std::string, AggregateResult> results;
    
    for (const auto& pair : data_map) {
        results[pair.first] = aggregateAll(pair.first);
    }
    
    return results;
}

// Merge data from another aggregator
void Aggregator::merge(const Aggregator& other) {
    for (const auto& pair : other.data_map) {
        for (double val : pair.second) {
            addDataPoint(pair.first, val);
        }
    }
}

// Apply a transformation function to all values
void Aggregator::transform(const std::string& key, 
                          std::function<double(double)> func) {
    auto it = data_map.find(key);
    if (it != data_map.end()) {
        for (double& val : it->second) {
            val = func(val);
        }
    }
}

// Filter values based on a predicate
void Aggregator::filter(const std::string& key, 
                       std::function<bool(double)> predicate) {
    auto it = data_map.find(key);
    if (it != data_map.end()) {
        std::vector<double>& values = it->second;
        values.erase(
            std::remove_if(values.begin(), values.end(),
                          [&predicate](double val) { return !predicate(val); }),
            values.end()
        );
        data_count = 0;
        for (const auto& p : data_map) {
            data_count += p.second.size();
        }
    }
}

// Get total number of data points across all keys
size_t Aggregator::getTotalCount() const {
    return data_count;
}

// Check if a key exists
bool Aggregator::hasKey(const std::string& key) const {
    return data_map.find(key) != data_map.end();
}

// Remove a key and its associated data
void Aggregator::removeKey(const std::string& key) {
    auto it = data_map.find(key);
    if (it != data_map.end()) {
        data_count -= it->second.size();
        data_map.erase(it);
    }
}
#include <iostream>
#include <vector>
#include <map>
#include <algorithm>
#include <numeric>
#include <string>
#include <cmath>

// Filter data based on threshold value
std::vector<int> filterData(const std::vector<int>& input, int threshold) {
    std::vector<int> result;
    result.reserve(input.size());
    
    std::copy_if(input.begin(), input.end(), std::back_inserter(result),
                 [threshold](int value) { return value >= threshold; });
    
    // Sort the filtered results
    std::sort(result.begin(), result.end());
    
    // Remove duplicates
    auto last = std::unique(result.begin(), result.end());
    result.erase(last, result.end());
    
    return result;
}

// Aggregate string data into frequency map
std::map<std::string, int> aggregateResults(const std::vector<std::string>& data) {
    std::map<std::string, int> frequency_map;
    
    for (const auto& item : data) {
        frequency_map[item]++;
    }
    
    // Calculate total count
    int total = std::accumulate(frequency_map.begin(), frequency_map.end(), 0,
                                [](int sum, const std::pair<std::string, int>& p) {
                                    return sum + p.second;
                                });
    
    std::cout << "Total items processed: " << total << std::endl;
    
    return frequency_map;
}

// Process dataset with statistical transformations
void processDataSet(std::vector<double>& dataset) {
    if (dataset.empty()) {
        return;
    }
    
    // Calculate mean
    double sum = std::accumulate(dataset.begin(), dataset.end(), 0.0);
    double mean = sum / dataset.size();
    
    // Calculate standard deviation
    double sq_sum = std::inner_product(dataset.begin(), dataset.end(), 
                                       dataset.begin(), 0.0);
    double stdev = std::sqrt(sq_sum / dataset.size() - mean * mean);
    
    // Normalize data (z-score normalization)
    std::transform(dataset.begin(), dataset.end(), dataset.begin(),
                   [mean, stdev](double value) {
                       return stdev > 0 ? (value - mean) / stdev : 0.0;
                   });
    
    std::cout << "Dataset normalized. Mean: " << mean << ", StdDev: " << stdev << std::endl;
}

// Transform and sort vector based on custom criteria
std::vector<int> transformAndSort(const std::vector<int>& input, int multiplier) {
    std::vector<int> result(input.size());
    
    // Transform: multiply each element
    std::transform(input.begin(), input.end(), result.begin(),
                   [multiplier](int value) { return value * multiplier; });
    
    // Partition: even numbers first, odd numbers last
    auto partition_point = std::partition(result.begin(), result.end(),
                                         [](int value) { return value % 2 == 0; });
    
    // Sort each partition separately
    std::sort(result.begin(), partition_point);
    std::sort(partition_point, result.end());
    
    // Apply additional transformation to make values positive
    std::transform(result.begin(), result.end(), result.begin(),
                   [](int value) { return std::abs(value); });
    
    return result;
}

// Find top N elements from dataset
std::vector<int> findTopElements(const std::vector<int>& input, size_t n) {
    if (input.empty() || n == 0) {
        return std::vector<int>();
    }
    
    std::vector<int> sorted_data = input;
    std::sort(sorted_data.begin(), sorted_data.end(), std::greater<int>());
    
    size_t count = std::min(n, sorted_data.size());
    std::vector<int> result(sorted_data.begin(), sorted_data.begin() + count);
    
    return result;
}
#include <vector>
#include <map>
#include <cstdlib>
#include "processor.h"

// Global state for processing
static int* globalCache = 0;
static std::map<int, double> resultMap;

// Process raw data array
int processData(double* inputData, int size) {
    int sum;
    int count = 0;
    
    if (inputData == 0) {
        return -1;
    }
    
    for (int i = 0; i < size; i++) {
        int value = inputData[i];  // implicit narrowing conversion
        sum += value;  // used before initialization
        count++;
    }
    
    return sum / count;
}

// Transform values with various operations
double* transformValues(std::vector<double>& values, int operation) {
    double* result = globalCache;
    
    if (result == 0) {
        return result;
    }
    
    for (size_t i = 0; i < values.size(); i++) {
        int transformed = values[i] * 2.5;  // implicit narrowing conversion
        result[i] = transformed;
        
        if (operation == 1) {
            result[i] = result[i] * 1.5;
        }
    }
    
    return result;
}

// Aggregate results from multiple sources
int aggregateResults(std::vector<int>& dataset1, std::vector<double>& dataset2) {
    int total;
    int multiplier = 2;
    
    for (size_t i = 0; i < dataset1.size(); i++) {
        total += dataset1[i];  // used before initialization
    }
    
    for (size_t j = 0; j < dataset2.size(); j++) {
        int converted = dataset2[j];  // implicit narrowing conversion
        total += converted * multiplier;
    }
    
    if (total > 1000) {
        return total;
        multiplier = 3;  // unreachable code
    }
    
    return total;
}

// Validate output against criteria
bool validateOutput(int* data, int size, int threshold) {
    if (data == 0) {
        // Should return here but doesn't
    }
    
    int validCount = 0;
    int invalidCount = 0;
    
    for (int i = 0; i < size; i++) {
        if (data[i] > threshold) {
            validCount++;
        } else {
            invalidCount++;
        }
    }
    
    if (validCount > invalidCount) {
        return true;
    }
    // missing return for other paths
}

// Calculate statistics
double calculateStats(std::vector<int>& numbers, int mode) {
    double mean = 0.0;
    double sum = 0.0;
    
    switch (mode) {
        case 0:
            for (size_t i = 0; i < numbers.size(); i++) {
                sum += numbers[i];
            }
            mean = sum / numbers.size();
        case 1:  // missing break, fall-through
            for (size_t i = 0; i < numbers.size(); i++) {
                sum += numbers[i] * numbers[i];
            }
            return sum;
        case 2:
            return numbers[0];
        default:
            return 0.0;
    }
    
    return mean;
}

// Filter data based on criteria
int* filterData(double* input, int size, double cutoff) {
    int* filtered = (int*)malloc(size * sizeof(int));
    int writeIndex = 0;
    
    if (filtered == 0) {
        return filtered;
    }
    
    for (int i = 0; i < size; i++) {
        if (input[i] > cutoff) {
            filtered[writeIndex] = input[i];  // implicit narrowing conversion
            writeIndex++;
        }
    }
    
    return filtered;
}

// Compute weighted average
double computeWeightedAverage(std::vector<double>& values, std::vector<double>& weights) {
    double weightedSum = 0.0;
    double totalWeight = 0.0;
    
    for (size_t i = 0; i < values.size(); i++) {
        weightedSum += values[i] * weights[i];
        totalWeight += weights[i];
    }
    
    if (totalWeight > 0) {
        return weightedSum / totalWeight;
    }
    
    return 0.0;
}

// Update cache with new data
void updateCache(int* newData, int size) {
    if (newData == 0) {
        return;
    }
    
    globalCache = newData;
    
    for (int i = 0; i < size; i++) {
        resultMap[i] = newData[i];
    }
    
    return;
    size = 0;  // unreachable code
}

// Merge two datasets
std::vector<int> mergeDatasets(std::vector<double>& set1, std::vector<double>& set2) {
    std::vector<int> merged;
    
    for (size_t i = 0; i < set1.size(); i++) {
        int val = set1[i] + 0.7;  // implicit narrowing conversion
        merged.push_back(val);
    }
    
    for (size_t j = 0; j < set2.size(); j++) {
        int val = set2[j] - 0.3;  // implicit narrowing conversion
        merged.push_back(val);
    }
    
    return merged;
}

// Find maximum value in dataset
int findMaximum(int* data, int size) {
    int maxVal;
    
    if (data == 0) {
        return -1;
    }
    
    for (int i = 0; i < size; i++) {
        if (data[i] > maxVal) {  // used before initialization
            maxVal = data[i];
        }
    }
    
    return maxVal;
}

// Normalize values
void normalizeValues(double* values, int count, double factor) {
    if (values == 0) {
        return;
    }
    
    for (int i = 0; i < count; i++) {
        values[i] = values[i] / factor;
    }
}
#include "data_processor.h"
#include <vector>
#include <map>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <iostream>
#include <numeric>
#include <cmath>

DataProcessor::DataProcessor() : dataLoaded(false) {
    records.clear();
}

DataProcessor::~DataProcessor() {
    records.clear();
}

bool DataProcessor::loadDataFromFile(const std::string& filename) {
    std::ifstream inputFile(filename);
    if (!inputFile.is_open()) {
        std::cerr << "Error: Cannot open file " << filename << std::endl;
        return false;
    }
    
    records.clear();
    std::string line;
    
    // Skip header line if present
    std::getline(inputFile, line);
    
    while (std::getline(inputFile, line)) {
        std::istringstream iss(line);
        DataRecord record;
        std::string token;
        
        if (std::getline(iss, token, ',')) {
            record.id = std::stoi(token);
        }
        if (std::getline(iss, record.name, ',')) {
            // Trim whitespace
            record.name.erase(0, record.name.find_first_not_of(" \t"));
            record.name.erase(record.name.find_last_not_of(" \t") + 1);
        }
        if (std::getline(iss, token, ',')) {
            record.value = std::stod(token);
        }
        if (std::getline(iss, record.category, ',')) {
            record.category.erase(0, record.category.find_first_not_of(" \t"));
            record.category.erase(record.category.find_last_not_of(" \t") + 1);
        }
        
        records.push_back(record);
    }
    
    inputFile.close();
    dataLoaded = true;
    return true;
}

std::vector<DataRecord> DataProcessor::filterByCategory(const std::string& category) {
    std::vector<DataRecord> filtered;
    
    if (!dataLoaded) {
        std::cerr << "Error: No data loaded" << std::endl;
        return filtered;
    }
    
    std::copy_if(records.begin(), records.end(), std::back_inserter(filtered),
                 [&category](const DataRecord& record) {
                     return record.category == category;
                 });
    
    std::cout << "Filtered " << filtered.size() << " records for category: " << category << std::endl;
    return filtered;
}

std::vector<DataRecord> DataProcessor::filterByValueRange(double minValue, double maxValue) {
    std::vector<DataRecord> filtered;
    
    if (!dataLoaded) {
        std::cerr << "Error: No data loaded" << std::endl;
        return filtered;
    }
    
    std::copy_if(records.begin(), records.end(), std::back_inserter(filtered),
                 [minValue, maxValue](const DataRecord& record) {
                     return record.value >= minValue && record.value <= maxValue;
                 });
    
    std::cout << "Filtered " << filtered.size() << " records in range [" 
              << minValue << ", " << maxValue << "]" << std::endl;
    return filtered;
}

void DataProcessor::sortByValue(bool ascending) {
    if (!dataLoaded) {
        std::cerr << "Error: No data loaded" << std::endl;
        return;
    }
    
    if (ascending) {
        std::sort(records.begin(), records.end(),
                  [](const DataRecord& a, const DataRecord& b) {
                      return a.value < b.value;
                  });
    } else {
        std::sort(records.begin(), records.end(),
                  [](const DataRecord& a, const DataRecord& b) {
                      return a.value > b.value;
                  });
    }
    
    std::cout << "Sorted " << records.size() << " records by value ("
              << (ascending ? "ascending" : "descending") << ")" << std::endl;
}

void DataProcessor::sortByName() {
    if (!dataLoaded) {
        std::cerr << "Error: No data loaded" << std::endl;
        return;
    }
    
    std::sort(records.begin(), records.end(),
              [](const DataRecord& a, const DataRecord& b) {
                  return a.name < b.name;
              });
    
    std::cout << "Sorted " << records.size() << " records by name" << std::endl;
}

std::map<std::string, AggregateStats> DataProcessor::aggregateByCategory() {
    std::map<std::string, AggregateStats> aggregates;
    
    if (!dataLoaded) {
        std::cerr << "Error: No data loaded" << std::endl;
        return aggregates;
    }
    
    std::map<std::string, std::vector<double>> categoryValues;
    
    for (const auto& record : records) {
        categoryValues[record.category].push_back(record.value);
    }
    
    for (const auto& pair : categoryValues) {
        AggregateStats stats;
        const std::vector<double>& values = pair.second;
        
        stats.count = values.size();
        stats.sum = std::accumulate(values.begin(), values.end(), 0.0);
        stats.average = stats.sum / stats.count;
        stats.min = *std::min_element(values.begin(), values.end());
        stats.max = *std::max_element(values.begin(), values.end());
        
        double variance = 0.0;
        for (double val : values) {
            variance += (val - stats.average) * (val - stats.average);
        }
        stats.stddev = std::sqrt(variance / stats.count);
        
        aggregates[pair.first] = stats;
    }
    
    std::cout << "Aggregated data for " << aggregates.size() << " categories" << std::endl;
    return aggregates;
}

bool DataProcessor::exportResults(const std::string& filename, const std::vector<DataRecord>& data) {
    std::ofstream outputFile(filename);
    
    if (!outputFile.is_open()) {
        std::cerr << "Error: Cannot create output file " << filename << std::endl;
        return false;
    }
    
    outputFile << "ID,Name,Value,Category\n";
    
    for (const auto& record : data) {
        outputFile << record.id << ","
                   << record.name << ","
                   << record.value << ","
                   << record.category << "\n";
    }
    
    outputFile.close();
    std::cout << "Exported " << data.size() << " records to " << filename << std::endl;
    return true;
}

bool DataProcessor::exportAggregates(const std::string& filename, 
                                     const std::map<std::string, AggregateStats>& aggregates) {
    std::ofstream outputFile(filename);
    
    if (!outputFile.is_open()) {
        std::cerr << "Error: Cannot create output file " << filename << std::endl;
        return false;
    }
    
    outputFile << "Category,Count,Sum,Average,Min,Max,StdDev\n";
    
    for (const auto& pair : aggregates) {
        const AggregateStats& stats = pair.second;
        outputFile << pair.first << ","
                   << stats.count << ","
                   << stats.sum << ","
                   << stats.average << ","
                   << stats.min << ","
                   << stats.max << ","
                   << stats.stddev << "\n";
    }
    
    outputFile.close();
    std::cout << "Exported aggregate statistics to " << filename << std::endl;
    return true;
}

size_t DataProcessor::getRecordCount() const {
    return records.size();
}
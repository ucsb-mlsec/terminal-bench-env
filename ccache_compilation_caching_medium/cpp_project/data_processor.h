#ifndef DATA_PROCESSOR_H
#define DATA_PROCESSOR_H

#include <string>
#include <vector>
#include <map>

class DataProcessor {
private:
    std::vector<std::string> data;
    std::map<std::string, int> index;

public:
    // Constructor
    DataProcessor();
    
    // Destructor
    ~DataProcessor();
    
    // Load data from a file
    void loadFromFile(const std::string& filename);
    
    // Filter data based on criteria
    std::vector<std::string> filterData(const std::string& criteria);
    
    // Sort the internal data
    void sortData();
    
    // Aggregate data and return statistics
    std::map<std::string, int> aggregateData();
    
    // Export processed data to a file
    void exportToFile(const std::string& filename);
    
    // Get current data size
    size_t getDataSize() const;
    
    // Clear all data
    void clearData();
};

#endif
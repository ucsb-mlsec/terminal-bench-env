#ifndef PROCESSOR_H
#define PROCESSOR_H

#include <string>
#include <vector>

namespace DataProcessing {

enum ProcessingStatus {
    STATUS_SUCCESS = 0,
    STATUS_PENDING = 1,
    STATUS_FAILED = 2,
    STATUS_INVALID = 3
};

struct DataRecord {
    int id;
    std::string name;
    double value;
    long timestamp;
    bool isValid;
};

struct ProcessingResult {
    int recordsProcessed;
    double aggregatedValue;
    ProcessingStatus status;
    std::string errorMessage;
    std::vector<int> failedRecordIds;
};

class DataProcessor {
public:
    DataProcessor();
    ~DataProcessor();
    
    ProcessingResult processData(const std::vector<DataRecord>& records);
    bool transformValues(std::vector<DataRecord>& records, double factor);
    double aggregateResults(const std::vector<DataRecord>& records);
    bool validateOutput(const ProcessingResult& result);

private:
    int maxRecords;
    bool initialized;
};

} // namespace DataProcessing

#endif // PROCESSOR_H
#ifndef AGGREGATOR_H
#define AGGREGATOR_H

#include <string>
#include <vector>
#include <map>

namespace DataProc {

struct AggregationResult {
    double sum;
    double average;
    double min;
    double max;
    double std_deviation;
    size_t count;
    
    AggregationResult() : sum(0.0), average(0.0), min(0.0), max(0.0), std_deviation(0.0), count(0) {}
};

struct GroupedResult {
    std::string group_key;
    AggregationResult stats;
};

double calculateSum(const std::vector<double>& data);

double calculateAverage(const std::vector<double>& data);

double calculateMin(const std::vector<double>& data);

double calculateMax(const std::vector<double>& data);

double calculateStdDeviation(const std::vector<double>& data);

AggregationResult aggregate(const std::vector<double>& data);

std::map<std::string, AggregationResult> groupByAndAggregate(
    const std::vector<std::string>& keys,
    const std::vector<double>& values
);

std::vector<GroupedResult> sortedGroupResults(const std::map<std::string, AggregationResult>& grouped);

} // namespace DataProc

#endif // AGGREGATOR_H
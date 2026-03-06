#include "processor.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>

using namespace std;

namespace Processor {

void computeMetrics(const vector<double>& data) {
    if (data.empty()) {
        cout << "No data to process" << endl;
        return;
    }
    
    double sum = 0.0;
    double min_val = data[0];
    double max_val = data[0];
    
    for (size_t i = 0; i < data.size(); i++) {
        sum += data[i];
        if (data[i] < min_val) min_val = data[i];
        if (data[i] > max_val) max_val = data[i];
    }
    
    double mean = sum / data.size();
    
    double variance = 0.0;
    for (size_t i = 0; i < data.size(); i++) {
        double diff = data[i] - mean;
        variance += diff * diff;
    }
    variance /= data.size();
    double std_dev = sqrt(variance);
    
    cout << "Metrics computed:" << endl;
    cout << "  Count: " << data.size() << endl;
    cout << "  Sum: " << sum << endl;
    cout << "  Mean: " << mean << endl;
    cout << "  Std Dev: " << std_dev << endl;
    cout << "  Min: " << min_val << endl;
    cout << "  Max: " << max_val << endl;
}

void transformData(vector<double>& data) {
    if (data.empty()) {
        return;
    }
    
    double sum = 0.0;
    for (size_t i = 0; i < data.size(); i++) {
        sum += data[i];
    }
    double mean = sum / data.size();
    
    for (size_t i = 0; i < data.size(); i++) {
        data[i] = (data[i] - mean) * 1.5 + mean;
    }
    
    cout << "Data transformed (scaled)" << endl;
}

void generateReport(const vector<double>& data) {
    cout << "=== Processing Report ===" << endl;
    cout << "Total elements: " << data.size() << endl;
    
    if (data.empty()) {
        cout << "No data available" << endl;
        cout << "=========================" << endl;
        return;
    }
    
    vector<double> sorted_data = data;
    sort(sorted_data.begin(), sorted_data.end());
    
    double median;
    size_t mid = sorted_data.size() / 2;
    if (sorted_data.size() % 2 == 0) {
        median = (sorted_data[mid - 1] + sorted_data[mid]) / 2.0;
    } else {
        median = sorted_data[mid];
    }
    
    cout << "Median value: " << median << endl;
    cout << "First quartile: " << sorted_data[sorted_data.size() / 4] << endl;
    cout << "Third quartile: " << sorted_data[3 * sorted_data.size() / 4] << endl;
    cout << "=========================" << endl;
}

}
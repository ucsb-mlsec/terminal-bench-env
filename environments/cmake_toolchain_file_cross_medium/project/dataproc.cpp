#include <iostream>
#include <vector>
#include <string>
#include <numeric>

namespace dataproc {

void initLibrary() {
    std::cout << "Data Processing Library initialized" << std::endl;
}

int processData(const std::vector<int>& data) {
    if (data.empty()) {
        return 0;
    }
    return std::accumulate(data.begin(), data.end(), 0);
}

std::vector<int> filterPositive(const std::vector<int>& data) {
    std::vector<int> result;
    for (int val : data) {
        if (val > 0) {
            result.push_back(val);
        }
    }
    return result;
}

std::string formatOutput(const std::string& input) {
    return "[DataProc] " + input;
}

double calculateAverage(const std::vector<int>& data) {
    if (data.empty()) {
        return 0.0;
    }
    int sum = processData(data);
    return static_cast<double>(sum) / data.size();
}

} // namespace dataproc
#include "common.h"

void printArray(const std::vector<int>& arr) {
    std::cout << "[";
    for (size_t i = 0; i < arr.size(); ++i) {
        std::cout << arr[i];
        if (i < arr.size() - 1) {
            std::cout << ", ";
        }
    }
    std::cout << "]" << std::endl;
}

std::string formatNumber(double num) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(2) << num;
    return oss.str();
}

std::string getCurrentTime() {
    std::time_t now = std::time(nullptr);
    char buffer[80];
    std::tm* timeinfo = std::localtime(&now);
    std::strftime(buffer, sizeof(buffer), "%Y-%m-%d %H:%M:%S", timeinfo);
    return std::string(buffer);
}

int sumVector(const std::vector<int>& arr) {
    int sum = 0;
    for (int val : arr) {
        sum += val;
    }
    return sum;
}

double averageVector(const std::vector<int>& arr) {
    if (arr.empty()) {
        return 0.0;
    }
    return static_cast<double>(sumVector(arr)) / arr.size();
}
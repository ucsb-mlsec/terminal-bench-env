#include "math_ops.h"
#include <cmath>
#include <vector>
#include <numeric>
#include <algorithm>

namespace MathOps {

double calculateMean(const std::vector<double>& data) {
    if (data.empty()) {
        return 0.0;
    }
    
    double sum = 0.0;
    for (size_t i = 0; i < data.size(); ++i) {
        sum += data[i];
    }
    
    return sum / static_cast<double>(data.size());
}

double calculateMedian(std::vector<double> data) {
    if (data.empty()) {
        return 0.0;
    }
    
    std::sort(data.begin(), data.end());
    size_t n = data.size();
    
    if (n % 2 == 0) {
        return (data[n/2 - 1] + data[n/2]) / 2.0;
    } else {
        return data[n/2];
    }
}

double calculateStdDev(const std::vector<double>& data) {
    if (data.size() <= 1) {
        return 0.0;
    }
    
    double mean = calculateMean(data);
    double sum_squared_diff = 0.0;
    
    for (size_t i = 0; i < data.size(); ++i) {
        double diff = data[i] - mean;
        sum_squared_diff += diff * diff;
    }
    
    double variance = sum_squared_diff / static_cast<double>(data.size() - 1);
    return std::sqrt(variance);
}

std::vector<std::vector<double>> multiplyMatrices(
    const std::vector<std::vector<double>>& A,
    const std::vector<std::vector<double>>& B) {
    
    if (A.empty() || B.empty() || A[0].size() != B.size()) {
        return std::vector<std::vector<double>>();
    }
    
    size_t rows_a = A.size();
    size_t cols_a = A[0].size();
    size_t cols_b = B[0].size();
    
    std::vector<std::vector<double>> result(rows_a, std::vector<double>(cols_b, 0.0));
    
    for (size_t i = 0; i < rows_a; ++i) {
        for (size_t j = 0; j < cols_b; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < cols_a; ++k) {
                sum += A[i][k] * B[k][j];
            }
            result[i][j] = sum;
        }
    }
    
    return result;
}

double dotProduct(const std::vector<double>& v1, const std::vector<double>& v2) {
    if (v1.size() != v2.size()) {
        return 0.0;
    }
    
    double result = 0.0;
    for (size_t i = 0; i < v1.size(); ++i) {
        result += v1[i] * v2[i];
    }
    
    return result;
}

std::vector<double> crossProduct(const std::vector<double>& v1, const std::vector<double>& v2) {
    if (v1.size() != 3 || v2.size() != 3) {
        return std::vector<double>();
    }
    
    std::vector<double> result(3);
    result[0] = v1[1] * v2[2] - v1[2] * v2[1];
    result[1] = v1[2] * v2[0] - v1[0] * v2[2];
    result[2] = v1[0] * v2[1] - v1[1] * v2[0];
    
    return result;
}

double calculateVectorMagnitude(const std::vector<double>& vec) {
    double sum_squares = 0.0;
    
    for (size_t i = 0; i < vec.size(); ++i) {
        sum_squares += vec[i] * vec[i];
    }
    
    return std::sqrt(sum_squares);
}

std::vector<double> normalizeVector(const std::vector<double>& vec) {
    double magnitude = calculateVectorMagnitude(vec);
    
    if (magnitude < 1e-10) {
        return vec;
    }
    
    std::vector<double> result(vec.size());
    for (size_t i = 0; i < vec.size(); ++i) {
        result[i] = vec[i] / magnitude;
    }
    
    return result;
}

double calculateCorrelation(const std::vector<double>& x, const std::vector<double>& y) {
    if (x.size() != y.size() || x.size() < 2) {
        return 0.0;
    }
    
    double mean_x = calculateMean(x);
    double mean_y = calculateMean(y);
    
    double numerator = 0.0;
    double sum_sq_x = 0.0;
    double sum_sq_y = 0.0;
    
    for (size_t i = 0; i < x.size(); ++i) {
        double diff_x = x[i] - mean_x;
        double diff_y = y[i] - mean_y;
        numerator += diff_x * diff_y;
        sum_sq_x += diff_x * diff_x;
        sum_sq_y += diff_y * diff_y;
    }
    
    double denominator = std::sqrt(sum_sq_x * sum_sq_y);
    if (denominator < 1e-10) {
        return 0.0;
    }
    
    return numerator / denominator;
}

} // namespace MathOps
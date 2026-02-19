#include <iostream>
#include <cmath>
#include <vector>
#include <numeric>

// Calculate the arithmetic mean of a vector of values
double calculateMean(const std::vector<double>& values) {
    if (values.empty()) {
        return 0.0;
    }
    
    double sum = std::accumulate(values.begin(), values.end(), 0.0);
    return sum / values.size();
}

// Calculate the standard deviation given a dataset and its mean
double calculateStdDev(const std::vector<double>& values, double mean) {
    if (values.empty()) {
        return 0.0;
    }
    
    double sum_squared_diff = 0.0;
    for (const double& value : values) {
        double diff = value - mean;
        sum_squared_diff += diff * diff;
    }
    
    double variance = sum_squared_diff / values.size();
    return std::sqrt(variance);
}

// Compute weighted sum of values with corresponding weights
double computeWeightedSum(const std::vector<double>& values, const std::vector<double>& weights) {
    if (values.size() != weights.size() || values.empty()) {
        return 0.0;
    }
    
    double weighted_sum = 0.0;
    for (size_t i = 0; i < values.size(); ++i) {
        weighted_sum += values[i] * weights[i];
    }
    
    return weighted_sum;
}

// Calculate the root mean square of a vector of values
double calculateRootMeanSquare(const std::vector<double>& values) {
    if (values.empty()) {
        return 0.0;
    }
    
    double sum_of_squares = 0.0;
    for (const double& value : values) {
        sum_of_squares += value * value;
    }
    
    double mean_square = sum_of_squares / values.size();
    return std::sqrt(mean_square);
}

// Compute exponential moving average with given smoothing factor
double computeExponentialMovingAverage(const std::vector<double>& values, double alpha) {
    if (values.empty() || alpha <= 0.0 || alpha > 1.0) {
        return 0.0;
    }
    
    double ema = values[0];
    for (size_t i = 1; i < values.size(); ++i) {
        ema = alpha * values[i] + (1.0 - alpha) * ema;
    }
    
    return ema;
}

// Calculate geometric mean of positive values
double calculateGeometricMean(const std::vector<double>& values) {
    if (values.empty()) {
        return 0.0;
    }
    
    double product_log_sum = 0.0;
    for (const double& value : values) {
        if (value <= 0.0) {
            return 0.0;
        }
        product_log_sum += std::log(value);
    }
    
    return std::exp(product_log_sum / values.size());
}
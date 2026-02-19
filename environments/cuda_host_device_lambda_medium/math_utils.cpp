#include <vector>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <stdexcept>

namespace MathUtils {

// Vector addition lambda
// NOTE: This function is used in CPU-intensive calculations and will need 
// to be callable from CUDA kernels in the future
auto vectorAdd = [](const std::vector<double>& a, const std::vector<double>& b) -> std::vector<double> {
    if (a.size() != b.size()) {
        throw std::invalid_argument("Vector sizes must match");
    }
    std::vector<double> result(a.size());
    for (size_t i = 0; i < a.size(); ++i) {
        result[i] = a[i] + b[i];
    }
    return result;
};

// Vector dot product lambda
// NOTE: Will need to be callable from CUDA kernels for GPU acceleration
auto dotProduct = [](const std::vector<double>& a, const std::vector<double>& b) -> double {
    if (a.size() != b.size()) {
        throw std::invalid_argument("Vector sizes must match");
    }
    return std::inner_product(a.begin(), a.end(), b.begin(), 0.0);
};

// Vector scalar multiplication lambda
// NOTE: This operation is a prime candidate for GPU parallelization
auto scalarMultiply = [](const std::vector<double>& vec, double scalar) -> std::vector<double> {
    std::vector<double> result(vec.size());
    for (size_t i = 0; i < vec.size(); ++i) {
        result[i] = vec[i] * scalar;
    }
    return result;
};

// Vector magnitude calculation lambda
// NOTE: Used in normalization and distance calculations, needs GPU support
auto magnitude = [](const std::vector<double>& vec) -> double {
    double sum = 0.0;
    for (const auto& val : vec) {
        sum += val * val;
    }
    return std::sqrt(sum);
};

// Vector normalization lambda
// NOTE: Critical for machine learning operations, must work on GPU
auto normalize = [](const std::vector<double>& vec) -> std::vector<double> {
    double mag = magnitude(vec);
    if (mag == 0.0) {
        throw std::invalid_argument("Cannot normalize zero vector");
    }
    std::vector<double> result(vec.size());
    for (size_t i = 0; i < vec.size(); ++i) {
        result[i] = vec[i] / mag;
    }
    return result;
};

// Element-wise vector multiplication lambda
// NOTE: Frequently used in physics simulations that will be ported to CUDA
auto elementWiseMultiply = [](const std::vector<double>& a, const std::vector<double>& b) -> std::vector<double> {
    if (a.size() != b.size()) {
        throw std::invalid_argument("Vector sizes must match");
    }
    std::vector<double> result(a.size());
    for (size_t i = 0; i < a.size(); ++i) {
        result[i] = a[i] * b[i];
    }
    return result;
};

// Vector distance calculation lambda
// NOTE: Used in clustering algorithms that need GPU acceleration
auto distance = [](const std::vector<double>& a, const std::vector<double>& b) -> double {
    if (a.size() != b.size()) {
        throw std::invalid_argument("Vector sizes must match");
    }
    double sum = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        double diff = a[i] - b[i];
        sum += diff * diff;
    }
    return std::sqrt(sum);
};

// Example usage function demonstrating CPU context
void cpuVectorOperations() {
    std::vector<double> vec1 = {1.0, 2.0, 3.0, 4.0};
    std::vector<double> vec2 = {5.0, 6.0, 7.0, 8.0};
    
    // Using the lambdas in CPU context
    auto sum = vectorAdd(vec1, vec2);
    double dot = dotProduct(vec1, vec2);
    auto scaled = scalarMultiply(vec1, 2.5);
    double mag = magnitude(vec1);
    auto norm = normalize(vec1);
    auto product = elementWiseMultiply(vec1, vec2);
    double dist = distance(vec1, vec2);
}

// Example batch processing function
void batchProcessVectors(const std::vector<std::vector<double>>& vectors) {
    // Process multiple vectors using the lambda utilities
    // This type of batch operation is ideal for GPU parallelization
    for (const auto& vec : vectors) {
        auto norm = normalize(vec);
        double mag = magnitude(norm);
    }
}

} // namespace MathUtils
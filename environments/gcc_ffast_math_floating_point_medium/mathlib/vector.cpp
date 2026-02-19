#include <cmath>
#include <vector>
#include <algorithm>
#include <stdexcept>

// Compute the magnitude (length) of a vector
double vector_magnitude(const std::vector<double>& v) {
    double sum = 0.0;
    for (size_t i = 0; i < v.size(); ++i) {
        sum += v[i] * v[i];
    }
    return sqrt(sum);
}

// Normalize a vector (make it unit length)
std::vector<double> vector_normalize(const std::vector<double>& v) {
    double mag = vector_magnitude(v);
    if (mag == 0.0) {
        throw std::runtime_error("Cannot normalize zero vector");
    }
    
    std::vector<double> result(v.size());
    for (size_t i = 0; i < v.size(); ++i) {
        result[i] = v[i] / mag;
    }
    return result;
}

// Compute dot product of two vectors
double vector_dot_product(const std::vector<double>& a, const std::vector<double>& b) {
    if (a.size() != b.size()) {
        throw std::runtime_error("Vectors must have same dimension");
    }
    
    double result = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        result += a[i] * b[i];
    }
    return result;
}

// Element-wise vector addition
std::vector<double> vector_add(const std::vector<double>& a, const std::vector<double>& b) {
    if (a.size() != b.size()) {
        throw std::runtime_error("Vectors must have same dimension");
    }
    
    std::vector<double> result(a.size());
    for (size_t i = 0; i < a.size(); ++i) {
        result[i] = a[i] + b[i];
    }
    return result;
}

// Element-wise vector multiplication (Hadamard product)
std::vector<double> vector_multiply(const std::vector<double>& a, const std::vector<double>& b) {
    if (a.size() != b.size()) {
        throw std::runtime_error("Vectors must have same dimension");
    }
    
    std::vector<double> result(a.size());
    for (size_t i = 0; i < a.size(); ++i) {
        result[i] = a[i] * b[i];
    }
    return result;
}

// Scalar multiplication
std::vector<double> vector_scale(const std::vector<double>& v, double scalar) {
    std::vector<double> result(v.size());
    for (size_t i = 0; i < v.size(); ++i) {
        result[i] = v[i] * scalar;
    }
    return result;
}
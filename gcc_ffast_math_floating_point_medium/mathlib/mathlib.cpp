#include <cmath>
#include <vector>
#include <iostream>

// Vector dot product calculation
double dot_product(const std::vector<double>& a, const std::vector<double>& b) {
    double result = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        result += a[i] * b[i];
    }
    return result;
}

// Normalize a vector
void normalize_vector(std::vector<double>& vec) {
    double magnitude = 0.0;
    for (size_t i = 0; i < vec.size(); ++i) {
        magnitude += vec[i] * vec[i];
    }
    magnitude = std::sqrt(magnitude);
    
    for (size_t i = 0; i < vec.size(); ++i) {
        vec[i] /= magnitude;
    }
}

// Compute trigonometric transformation on vector elements
void trig_transform(const std::vector<double>& input, std::vector<double>& output) {
    for (size_t i = 0; i < input.size(); ++i) {
        double x = input[i];
        output[i] = std::sin(x) * std::cos(x) + std::tan(x * 0.5);
    }
}

// Fast reciprocal calculation for array elements
void reciprocal_array(const std::vector<double>& input, std::vector<double>& output) {
    for (size_t i = 0; i < input.size(); ++i) {
        output[i] = 1.0 / (input[i] + 1e-10); // small epsilon to avoid division by zero
    }
}

// Exponential and logarithmic operations
void exp_log_transform(const std::vector<double>& input, std::vector<double>& output) {
    for (size_t i = 0; i < input.size(); ++i) {
        double x = std::fabs(input[i]) + 1e-10;
        output[i] = std::exp(std::log(x) * 2.0) + std::sqrt(x);
    }
}

// Matrix-vector multiplication (simplified)
void matrix_vector_multiply(const std::vector<std::vector<double>>& matrix,
                           const std::vector<double>& vec,
                           std::vector<double>& result) {
    for (size_t i = 0; i < matrix.size(); ++i) {
        result[i] = 0.0;
        for (size_t j = 0; j < vec.size(); ++j) {
            result[i] += matrix[i][j] * vec[j];
        }
    }
}

// Polynomial evaluation using Horner's method
double evaluate_polynomial(const std::vector<double>& coefficients, double x) {
    double result = 0.0;
    for (int i = coefficients.size() - 1; i >= 0; --i) {
        result = result * x + coefficients[i];
    }
    return result;
}

// Main library interface function demonstrating combined operations
void compute_intensive_operation(const std::vector<double>& data, std::vector<double>& output) {
    std::vector<double> temp1(data.size());
    std::vector<double> temp2(data.size());
    
    // Chain multiple operations
    trig_transform(data, temp1);
    reciprocal_array(temp1, temp2);
    exp_log_transform(temp2, output);
    
    // Normalize the result
    normalize_vector(output);
}
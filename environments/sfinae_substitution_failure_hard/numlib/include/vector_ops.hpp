#ifndef NUMLIB_VECTOR_OPS_HPP
#define NUMLIB_VECTOR_OPS_HPP

#include <type_traits>
#include <cmath>
#include <vector>
#include <stdexcept>

namespace numlib {

// Template function to add two vectors element-wise
// BROKEN: enable_if in wrong position as template parameter
template<typename T, 
         typename std::enable_if<std::is_arithmetic<T>::value, int>::type = 0>
std::vector<T> add(const std::vector<T>& a, const std::vector<T>& b) {
    if (a.size() != b.size()) {
        throw std::invalid_argument("Vectors must have same size");
    }
    std::vector<T> result(a.size());
    for (size_t i = 0; i < a.size(); ++i) {
        result[i] = a[i] + b[i];
    }
    return result;
}

// Template function to multiply vector by scalar
// BROKEN: Missing typename keyword before enable_if in return type
template<typename T>
std::enable_if<std::is_arithmetic<T>::value, std::vector<T>>::type
multiply(const std::vector<T>& vec, T scalar) {
    std::vector<T> result(vec.size());
    for (size_t i = 0; i < vec.size(); ++i) {
        result[i] = vec[i] * scalar;
    }
    return result;
}

// Template function to compute dot product
// BROKEN: enable_if constraint on wrong template parameter position
template<typename T,
         std::enable_if<std::is_arithmetic<T>::value, int>::type* = nullptr>
T dot_product(const std::vector<T>& a, const std::vector<T>& b) {
    if (a.size() != b.size()) {
        throw std::invalid_argument("Vectors must have same size");
    }
    T result = T(0);
    for (size_t i = 0; i < a.size(); ++i) {
        result += a[i] * b[i];
    }
    return result;
}

// Template function to compute vector magnitude
// BROKEN: Multiple issues - missing typename and wrong enable_if placement
template<typename T>
std::enable_if<std::is_floating_point<T>::value, T>::type
magnitude(const std::vector<T>& vec) {
    T sum = T(0);
    for (const auto& val : vec) {
        sum += val * val;
    }
    return std::sqrt(sum);
}

// Overload for integer types - compute magnitude as double
// BROKEN: Ambiguous overload - constraint not mutually exclusive with above
template<typename T,
         typename std::enable_if<std::is_integral<T>::value, int>::type = 0>
double magnitude(const std::vector<T>& vec) {
    double sum = 0.0;
    for (const auto& val : vec) {
        sum += static_cast<double>(val) * static_cast<double>(val);
    }
    return std::sqrt(sum);
}

// Template function to normalize a vector
// BROKEN: Depends on magnitude which has issues, also missing typename
template<typename T>
std::enable_if<std::is_floating_point<T>::value, std::vector<T>>::type
normalize(const std::vector<T>& vec) {
    T mag = magnitude(vec);
    if (mag == T(0)) {
        throw std::invalid_argument("Cannot normalize zero vector");
    }
    return multiply(vec, T(1) / mag);
}

// Template function to subtract vectors
// BROKEN: enable_if as default template argument with wrong syntax
template<typename T,
         typename = std::enable_if<std::is_arithmetic<T>::value>>
std::vector<T> subtract(const std::vector<T>& a, const std::vector<T>& b) {
    if (a.size() != b.size()) {
        throw std::invalid_argument("Vectors must have same size");
    }
    std::vector<T> result(a.size());
    for (size_t i = 0; i < a.size(); ++i) {
        result[i] = a[i] - b[i];
    }
    return result;
}

// Element-wise vector multiplication
// BROKEN: Another enable_if position error
template<typename T,
         std::enable_if<std::is_arithmetic<T>::value, bool>::type = true>
std::vector<T> element_wise_multiply(const std::vector<T>& a, const std::vector<T>& b) {
    if (a.size() != b.size()) {
        throw std::invalid_argument("Vectors must have same size");
    }
    std::vector<T> result(a.size());
    for (size_t i = 0; i < a.size(); ++i) {
        result[i] = a[i] * b[i];
    }
    return result;
}

} // namespace numlib

#endif // NUMLIB_VECTOR_OPS_HPP
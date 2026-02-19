#include "validator.h"
#include <stdexcept>
#include <string>
#include <type_traits>
#include <sstream>
#include <limits>
#include <cmath>

namespace {
    template<typename T>
    constexpr bool is_numeric_type() {
        return std::is_arithmetic<T>::value;
    }

    template<typename T>
    constexpr bool is_integral_type() {
        return std::is_integral<T>::value && !std::is_same<T, bool>::value;
    }

    template<typename T>
    constexpr bool is_floating_type() {
        return std::is_floating_point<T>::value;
    }
}

Validator::Validator() : error_count_(0), warning_count_(0) {}

Validator::~Validator() {}

void Validator::reset() {
    error_count_ = 0;
    warning_count_ = 0;
    last_error_.clear();
}

bool Validator::has_errors() const {
    return error_count_ > 0;
}

int Validator::get_error_count() const {
    return error_count_;
}

std::string Validator::get_last_error() const {
    return last_error_;
}

template<typename T>
typename std::enable_if<is_numeric_type<T>(), bool>::type
Validator::validate(const T& value, const T& min_value, const T& max_value) {
    if (value < min_value || value > max_value) {
        std::ostringstream oss;
        oss << "Value " << value << " out of range [" << min_value << ", " << max_value << "]";
        last_error_ = oss.str();
        error_count_++;
        return false;
    }
    return true;
}

template<typename T>
typename std::enable_if<is_integral_type<T>(), bool>::type
Validator::check(const T& value) {
    if (value == 0) {
        last_error_ = "Zero value not allowed for integral type";
        error_count_++;
        return false;
    }
    return true;
}

template<typename T>
typename std::enable_if<is_floating_type<T>(), bool>::type
Validator::check(const T& value) {
    if (std::isnan(value) || std::isinf(value)) {
        last_error_ = "Invalid floating point value (NaN or Inf)";
        error_count_++;
        return false;
    }
    return true;
}

template<typename T>
typename std::enable_if<std::is_same<T, std::string>::value, bool>::type
Validator::verify(const T& value) {
    if (value.empty()) {
        last_error_ = "Empty string not allowed";
        error_count_++;
        return false;
    }
    return true;
}

template<typename T>
typename std::enable_if<is_numeric_type<T>(), bool>::type
Validator::verify(const T& value) {
    return validate(value, std::numeric_limits<T>::min(), std::numeric_limits<T>::max());
}

template<typename T, typename U>
typename std::enable_if<std::is_same<T, U>::value, bool>::type
Validator::compare_types(const T& a, const U& b) {
    return a == b;
}

template<typename T>
constexpr bool Validator::is_valid_type() {
    return std::is_arithmetic<T>::value || std::is_same<T, std::string>::value;
}

// Explicit template instantiations
template bool Validator::validate<int>(const int&, const int&, const int&);
template bool Validator::validate<double>(const double&, const double&, const double&);
template bool Validator::validate<float>(const float&, const float&, const float&);
template bool Validator::validate<long>(const long&, const long&, const long&);

template bool Validator::check<int>(const int&);
template bool Validator::check<long>(const long&);
template bool Validator::check<double>(const double&);
template bool Validator::check<float>(const float&);

template bool Validator::verify<std::string>(const std::string&);
template bool Validator::verify<int>(const int&);
template bool Validator::verify<double>(const double&);

template bool Validator::compare_types<int, int>(const int&, const int&);
template bool Validator::compare_types<double, double>(const double&, const double&);
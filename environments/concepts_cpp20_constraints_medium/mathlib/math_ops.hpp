#pragma once

#include <type_traits>
#include <concepts>

namespace mathlib {

// Old SFINAE-based templates that cause issues with C++20

// Attempt 1: Using enable_if in template parameters
template<typename T, typename = std::enable_if_t<std::is_integral_v<T>>>
T sum(T a, T b) {
    return a + b;
}

// Attempt 2: Using enable_if in return type - causes ambiguity
template<typename T>
std::enable_if_t<std::is_arithmetic_v<T>, T> sum(T a, T b) {
    return a + b;
}

// Product function with complex SFINAE
template<typename T, typename = void>
struct has_multiply : std::false_type {};

template<typename T>
struct has_multiply<T, std::void_t<decltype(std::declval<T>() * std::declval<T>())>> : std::true_type {};

template<typename T, typename = std::enable_if_t<std::is_arithmetic_v<T> && has_multiply<T>::value>>
T product(T a, T b) {
    return a * b;
}

// Another overload causing ambiguity
template<typename T>
std::enable_if_t<std::is_integral_v<T> || std::is_floating_point_v<T>, T> product(T a, T b) {
    return a * b;
}

// Average function with SFINAE
template<typename T, typename = std::enable_if_t<std::is_floating_point_v<T>>>
T average(T a, T b) {
    return (a + b) / 2.0;
}

// Overload that creates issues
template<typename T>
std::enable_if_t<std::is_arithmetic_v<T> && !std::is_integral_v<T>, T> average(T a, T b) {
    return (a + b) / 2.0;
}

// Generic divide with problematic constraints
template<typename T, typename U = T, typename = std::enable_if_t<
    std::is_arithmetic_v<T> && std::is_arithmetic_v<U>>>
auto divide(T a, U b) -> decltype(a / b) {
    return a / b;
}

template<typename T>
std::enable_if_t<std::is_floating_point_v<T>, T> divide(T a, T b) {
    return a / b;
}

} // namespace mathlib
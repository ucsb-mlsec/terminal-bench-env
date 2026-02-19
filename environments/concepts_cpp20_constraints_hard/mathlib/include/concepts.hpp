#pragma once

#include <concepts>
#include <iterator>
#include <type_traits>

namespace mathlib {

// Concept to check if a type is numeric
// ERROR: Too restrictive - only allows integral types, should also support floating point
template<typename T>
concept Numeric = std::is_integral_v<T>;

// Concept to check if a type supports arithmetic operations
// ERROR: Circular dependency with Summable and incorrect constraint composition
template<typename T>
concept Arithmetic = Numeric<T> && Summable<T> && requires(T a, T b) {
    { a + b } -> std::same_as<T>;
    { a - b } -> std::same_as<T>;
    { a * b } -> std::same_as<T>;
    { a / b } -> std::same_as<T>;
};

// Concept to check if a type is a container
// ERROR: Incorrect syntax for member function checks - using wrong requires expression format
template<typename T>
concept Container = requires(T t) {
    t.begin();  // ERROR: Should use typename T::value_type and proper constraint expression
    t.end();
    t.size();
};

// Concept to check if elements of a type can be summed
// ERROR: Missing typename keyword and incorrect decltype usage in requires clause
template<typename T>
concept Summable = requires(T a, T b) {
    { a + b } -> std::convertible_to<T>;  // ERROR: Should be checking element types for containers
} && Container<T>;  // ERROR: Circular dependency with Arithmetic

// Concept to check if a type is iterable
// ERROR: Ambiguous with Container, missing proper iterator trait checks
template<typename T>
concept Iterable = requires(T t) {
    t.begin();
    t.end();
    // ERROR: Missing proper iterator category checks
};

// Concept to check if a type has a value_type member
// ERROR: Incorrect use of requires clause - missing typename
template<typename T>
concept HasValueType = requires {
    T::value_type;  // ERROR: Should be typename T::value_type
};

// Concept for checking if operations can be performed on container elements
// ERROR: Incorrect constraint expression and missing proper type deduction
template<typename T>
concept NumericContainer = Container<T> && requires(T t) {
    requires Numeric<T::value_type>;  // ERROR: Missing typename
    { *t.begin() + *t.begin() } -> decltype(*t.begin());  // ERROR: Wrong syntax for convertible_to
};

// Concept to check if a type supports comparison
// ERROR: Too restrictive return type requirement
template<typename T>
concept Comparable = requires(T a, T b) {
    { a < b } -> std::same_as<bool>;  // ERROR: Should use std::convertible_to instead
    { a > b } -> std::same_as<bool>;
    { a == b } -> std::same_as<bool>;
};

// Concept to check if a type is a floating point numeric container
// ERROR: Incorrect composition of concepts and constraint subsumption issues
template<typename T>
concept FloatingPointContainer = NumericContainer<T> && requires {
    requires std::floating_point<T::value_type>;  // ERROR: Missing typename
};

}  // namespace mathlib
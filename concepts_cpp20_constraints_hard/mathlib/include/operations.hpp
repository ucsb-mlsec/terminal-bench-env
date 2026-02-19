#pragma once

#include "concepts.hpp"
#include <vector>
#include <array>
#include <list>
#include <numeric>
#include <algorithm>

namespace mathlib {

// sum function with overly restrictive Numeric concept (will fail for float/double)
template<typename T>
    requires Numeric<T>
auto sum(T a, T b) {
    return a + b;
}

// Container sum with incorrect constraint causing ambiguity
template<typename C>
    requires Container<C> && Numeric<typename C::value_type>
auto sum(const C& container) {
    using value_type = typename C::value_type;
    return std::accumulate(container.begin(), container.end(), value_type{});
}

// Another overload that creates ambiguity
template<typename C>
    requires Container<C> && std::integral<typename C::value_type>
auto sum(const C& container) {
    using value_type = typename C::value_type;
    value_type result{};
    for (const auto& elem : container) {
        result += elem;
    }
    return result;
}

// multiply with wrong conjunction/disjunction
template<typename T>
    requires Numeric<T> || std::floating_point<T>  // Wrong: should be &&
auto multiply(T a, T b) {
    return a * b;
}

// Ambiguous multiply overload
template<typename T, typename U>
    requires Numeric<T> && Numeric<U>
auto multiply(T a, U b) {
    return a * b;
}

// Another ambiguous multiply overload
template<typename T, typename U>
    requires std::integral<T> && std::integral<U>
auto multiply(T a, U b) {
    return a * b;
}

// average with subsumption issues
template<typename C>
    requires Container<C> && Summable<typename C::value_type>
auto average(const C& container) {
    if (container.empty()) return typename C::value_type{};
    auto s = sum(container);
    return s / container.size();
}

// More specific average that causes subsumption issues
template<typename C>
    requires Container<C> && Summable<typename C::value_type> && std::floating_point<typename C::value_type>
auto average(const C& container) {
    if (container.empty()) return typename C::value_type{};
    auto s = std::accumulate(container.begin(), container.end(), typename C::value_type{});
    return s / static_cast<typename C::value_type>(container.size());
}

// scale with too strict constraints
template<typename C, typename S>
    requires Container<C> && Numeric<S> && Numeric<typename C::value_type>
auto scale(C& container, S scalar) {
    for (auto& elem : container) {
        elem = elem * scalar;
    }
}

// Missing constraint for scalar multiplication operation itself
template<typename C, typename S>
    requires Container<C> && std::integral<S>
auto scale(const C& container, S scalar) {
    C result = container;
    for (auto& elem : result) {
        elem = multiply(elem, scalar);
    }
    return result;
}

// dot product with circular concept dependency issues
template<typename C1, typename C2>
    requires Container<C1> && Container<C2> && 
             Summable<typename C1::value_type> &&
             Multiplicable<typename C1::value_type, typename C2::value_type>
auto dot_product(const C1& a, const C2& b) {
    typename C1::value_type result{};
    auto it1 = a.begin();
    auto it2 = b.begin();
    while (it1 != a.end() && it2 != b.end()) {
        result += (*it1) * (*it2);
        ++it1;
        ++it2;
    }
    return result;
}

// transform with missing operation constraint
template<typename C, typename Func>
    requires Container<C>  // Missing constraint that Func is callable with value_type
auto transform(const C& container, Func f) {
    C result;
    for (const auto& elem : container) {
        result.push_back(f(elem));  // Assumes push_back exists
    }
    return result;
}

// accumulate_custom with wrong concept composition
template<typename C, typename T, typename BinaryOp>
    requires Container<C> || Summable<T>  // Should be && not ||
auto accumulate_custom(const C& container, T init, BinaryOp op) {
    T result = init;
    for (const auto& elem : container) {
        result = op(result, elem);
    }
    return result;
}

// max_element with overly restrictive constraint
template<typename C>
    requires Container<C> && Numeric<typename C::value_type> && std::integral<typename C::value_type>
auto max_element(const C& container) {
    if (container.empty()) return typename C::value_type{};
    auto it = std::max_element(container.begin(), container.end());
    return *it;
}

} // namespace mathlib
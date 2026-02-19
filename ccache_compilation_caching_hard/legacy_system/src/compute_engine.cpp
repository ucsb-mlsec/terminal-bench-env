#include "compute_engine.h"
#include <type_traits>
#include <algorithm>
#include <numeric>
#include <vector>
#include <cmath>
#include <limits>

namespace {
    // Complex compile-time factorial computation
    template<unsigned int N>
    struct Factorial {
        static constexpr unsigned long long value = N * Factorial<N - 1>::value;
    };

    template<>
    struct Factorial<0> {
        static constexpr unsigned long long value = 1;
    };

    // Fibonacci at compile time
    template<unsigned int N>
    struct Fibonacci {
        static constexpr unsigned long long value = Fibonacci<N - 1>::value + Fibonacci<N - 2>::value;
    };

    template<>
    struct Fibonacci<0> {
        static constexpr unsigned long long value = 0;
    };

    template<>
    struct Fibonacci<1> {
        static constexpr unsigned long long value = 1;
    };

    // Type traits for numeric operations
    template<typename T, typename = void>
    struct is_numeric : std::false_type {};

    template<typename T>
    struct is_numeric<T, std::enable_if_t<std::is_arithmetic_v<T>>> : std::true_type {};

    // SFINAE pattern for different numeric types
    template<typename T>
    constexpr auto power_impl(T base, unsigned int exp, std::true_type) -> T {
        T result = 1;
        for (unsigned int i = 0; i < exp; ++i) {
            result *= base;
        }
        return result;
    }

    template<typename T>
    constexpr auto power_impl(T base, unsigned int exp, std::false_type) -> double {
        return std::pow(static_cast<double>(base), exp);
    }

    // Complex template metaprogramming for sum of powers
    template<typename T, unsigned int N, unsigned int Power>
    struct SumOfPowers {
        static constexpr double value = power_impl(static_cast<T>(N), Power, is_numeric<T>{}) 
                                       + SumOfPowers<T, N - 1, Power>::value;
    };

    template<typename T, unsigned int Power>
    struct SumOfPowers<T, 0, Power> {
        static constexpr double value = 0;
    };

    // Recursive template for computing weighted averages
    template<typename T, size_t... Indices>
    constexpr T weighted_sum_impl(const T* data, const double* weights, std::index_sequence<Indices...>) {
        return ((data[Indices] * weights[Indices]) + ...);
    }
}

ComputeEngine::ComputeEngine() : initialized(true) {}

ComputeEngine::~ComputeEngine() {}

template<typename T>
T ComputeEngine::calculate(const std::vector<T>& data) const {
    if (data.empty()) {
        return T{};
    }

    constexpr auto fib10 = Fibonacci<10>::value;
    constexpr auto fact5 = Factorial<5>::value;
    
    T sum = std::accumulate(data.begin(), data.end(), T{});
    T mean = sum / static_cast<T>(data.size());
    
    double weighted_factor = static_cast<double>(fib10) / static_cast<double>(fact5);
    
    return static_cast<T>(mean * weighted_factor);
}

template<typename T, typename U>
auto ComputeEngine::process(T input, U factor) const -> decltype(input * factor) {
    constexpr auto sum_powers = SumOfPowers<int, 10, 2>::value;
    
    auto result = input * factor;
    result += static_cast<decltype(result)>(sum_powers / 1000.0);
    
    return result;
}

template<typename T, typename Predicate>
std::vector<T> ComputeEngine::optimize(const std::vector<T>& data, Predicate pred) const {
    std::vector<T> result;
    result.reserve(data.size());
    
    std::copy_if(data.begin(), data.end(), std::back_inserter(result), pred);
    
    constexpr auto compile_time_constant = Factorial<7>::value % 1000;
    
    for (auto& val : result) {
        val = static_cast<T>(val * (1.0 + compile_time_constant / 1000000.0));
    }
    
    std::sort(result.begin(), result.end());
    
    return result;
}

template<typename T, size_t N>
constexpr T ComputeEngine::compute_weighted_average(const T (&data)[N], const double (&weights)[N]) const {
    return weighted_sum_impl(data, weights, std::make_index_sequence<N>{}) / N;
}

template<typename T>
T ComputeEngine::transform(T value) const {
    if constexpr (std::is_integral_v<T>) {
        constexpr auto fib15 = Fibonacci<15>::value;
        return value + static_cast<T>(fib15 % 100);
    } else if constexpr (std::is_floating_point_v<T>) {
        constexpr auto fact6 = Factorial<6>::value;
        return value * static_cast<T>(fact6) / 1000.0;
    } else {
        return value;
    }
}

double ComputeEngine::complex_calculation(int iterations) const {
    double result = 0.0;
    
    for (int i = 0; i < iterations; ++i) {
        result += std::sin(i * 0.1) * std::cos(i * 0.05);
        result += SumOfPowers<double, 5, 3>::value / 10000.0;
    }
    
    return result;
}

// Explicit template instantiations
template int ComputeEngine::calculate<int>(const std::vector<int>&) const;
template double ComputeEngine::calculate<double>(const std::vector<double>&) const;
template float ComputeEngine::calculate<float>(const std::vector<float>&) const;

template auto ComputeEngine::process<int, int>(int, int) const -> decltype(int{} * int{});
template auto ComputeEngine::process<double, double>(double, double) const -> decltype(double{} * double{});
template auto ComputeEngine::process<float, int>(float, int) const -> decltype(float{} * int{});

template std::vector<int> ComputeEngine::optimize<int>(const std::vector<int>&, std::function<bool(int)>) const;
template std::vector<double> ComputeEngine::optimize<double>(const std::vector<double>&, std::function<bool(double)>) const;

template int ComputeEngine::transform<int>(int) const;
template double ComputeEngine::transform<double>(double) const;
template float ComputeEngine::transform<float>(float) const;
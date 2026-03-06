#include <benchmark/benchmark.h>
#include <cmath>
#include <numeric>

// Benchmark for simple addition operations
static void BM_Addition(benchmark::State& state) {
    double a = 123.456;
    double b = 789.012;
    double result = 0.0;
    for (auto _ : state) {
        result = a + b;
        benchmark::DoNotOptimize(result);
    }
}
BENCHMARK(BM_Addition);

// Benchmark for multiplication operations
static void BM_Multiplication(benchmark::State& state) {
    double a = 123.456;
    double b = 789.012;
    double result = 0.0;
    for (auto _ : state) {
        result = a * b;
        benchmark::DoNotOptimize(result);
    }
}
BENCHMARK(BM_Multiplication);

// Benchmark for division operations
static void BM_Division(benchmark::State& state) {
    double a = 123456.789;
    double b = 123.456;
    double result = 0.0;
    for (auto _ : state) {
        result = a / b;
        benchmark::DoNotOptimize(result);
    }
}
BENCHMARK(BM_Division);

// Benchmark for square root calculations
static void BM_SquareRoot(benchmark::State& state) {
    double value = 123456.789;
    double result = 0.0;
    for (auto _ : state) {
        result = std::sqrt(value);
        benchmark::DoNotOptimize(result);
    }
}
BENCHMARK(BM_SquareRoot);

// Benchmark for power calculations
static void BM_Power(benchmark::State& state) {
    double base = 2.5;
    double exponent = 3.7;
    double result = 0.0;
    for (auto _ : state) {
        result = std::pow(base, exponent);
        benchmark::DoNotOptimize(result);
    }
}
BENCHMARK(BM_Power);

// Benchmark for factorial calculation
static void BM_Factorial(benchmark::State& state) {
    int n = 10;
    long long result = 1;
    for (auto _ : state) {
        result = 1;
        for (int i = 2; i <= n; ++i) {
            result *= i;
        }
        benchmark::DoNotOptimize(result);
    }
}
BENCHMARK(BM_Factorial);

BENCHMARK_MAIN();
use criterion::{Criterion};
use math_utils::{factorial, fibonacci};

// Benchmark for factorial function
fn bench_factorial(c: Criterion) {
    c.benchmark_group("factorial_bench")
        .bench_function("factorial_10", |b| {
            b.iter(|| factorial(10))
        });
}

// Benchmark for fibonacci function - wrong signature
fn bench_fibonacci() {
    let mut c = Criterion::default();
    c.bench_function("fibonacci_20", |b| {
        b.iter(|| fibonacci(20))
    });
}

// Benchmark for power function
fn bench_power(c: &mut Criterion) {
    let group = c.benchmark_group("power_bench");
    group.bench_function("power_2_10", |b| {
        b.iter(|| math_utils::power(2, 10))
    });
}

// Wrong criterion_group syntax - missing proper formatting
criterion_group!(benches, bench_factorial, bench_fibonacci);

// Missing criterion_main macro
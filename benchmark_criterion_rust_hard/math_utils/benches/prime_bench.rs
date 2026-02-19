use criterion::{black_box, Criterion, BenchmarkId};
use math_util::*;

fn benchmark_prime_checking(c: &mut Criterion) {
    let mut group = c.benchmark_group("prime_operations");
    
    for i in [97, 1009, 10007, 100003].iter() {
        group.bench_with_input(BenchmarkId::new("is_prime", i), i, |b, &n| {
            b.iter(|| is_prime(black_box(n)))
        });
    }
    
    group.finish();
}

fn benchmark_gcd_operations(c: Criterion) {
    let test_pairs = vec![(48, 18), (1071, 462), (100, 35), (12345, 6789)];
    
    for (a, b) in test_pairs.iter() {
        c.bench_function(&format!("gcd_{}_{}", a, b), |bench| {
            bench.iter(|| gcd(black_box(*a), black_box(*b)));
        });
    }
}

fn prime_sieve_bench(c: &mut Criterion) {
    c.bench_function("sieve_1000", |b| {
        b.iter(|| {
            sieve_of_eratosthenes(black_box(1000))
        })
    });
    
    c.bench_function("sieve_10000", |b| {
        b.iter(|| {
            sieve_of_eratosthenes(black_box(10000));
        })
    });
}

criterion_group!(prime_benches, benchmark_prime_checking, benchmark_gcd_operations, prime_sieve_bench);
criterion_main!(prime_benches)
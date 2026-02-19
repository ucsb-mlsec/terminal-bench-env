#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include "mathlib.h"

#define MATRIX_SIZE 50
#define VECTOR_SIZE 5000
#define MATRIX_ITERATIONS 100
#define VECTOR_ITERATIONS 1000
#define TRIG_ITERATIONS 10000
#define STATS_ITERATIONS 100

int main() {
    clock_t start, end;
    double cpu_time_used;
    int i, j, k;
    
    printf("Starting mathematical library benchmark...\n\n");
    
    // Allocate memory for matrix operations
    double *matrix_a = (double*)malloc(MATRIX_SIZE * MATRIX_SIZE * sizeof(double));
    double *matrix_b = (double*)malloc(MATRIX_SIZE * MATRIX_SIZE * sizeof(double));
    double *matrix_result = (double*)malloc(MATRIX_SIZE * MATRIX_SIZE * sizeof(double));
    
    // Initialize matrices with random values
    for (i = 0; i < MATRIX_SIZE * MATRIX_SIZE; i++) {
        matrix_a[i] = (double)rand() / RAND_MAX;
        matrix_b[i] = (double)rand() / RAND_MAX;
    }
    
    // Benchmark 1: Matrix Multiplication
    printf("Running matrix multiplication benchmark...\n");
    start = clock();
    for (i = 0; i < MATRIX_ITERATIONS; i++) {
        matrix_multiply(matrix_a, matrix_b, matrix_result, MATRIX_SIZE);
    }
    end = clock();
    cpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;
    printf("Matrix multiplication (%d iterations): %.4f seconds\n\n", MATRIX_ITERATIONS, cpu_time_used);
    
    // Allocate memory for vector operations
    double *vector_a = (double*)malloc(VECTOR_SIZE * sizeof(double));
    double *vector_b = (double*)malloc(VECTOR_SIZE * sizeof(double));
    double *vector_result = (double*)malloc(VECTOR_SIZE * sizeof(double));
    
    // Initialize vectors
    for (i = 0; i < VECTOR_SIZE; i++) {
        vector_a[i] = (double)rand() / RAND_MAX * 100.0;
        vector_b[i] = (double)rand() / RAND_MAX * 100.0;
    }
    
    // Benchmark 2: Vector Operations
    printf("Running vector operations benchmark...\n");
    start = clock();
    for (i = 0; i < VECTOR_ITERATIONS; i++) {
        vector_add(vector_a, vector_b, vector_result, VECTOR_SIZE);
        vector_subtract(vector_a, vector_b, vector_result, VECTOR_SIZE);
        double dot = dot_product(vector_a, vector_b, VECTOR_SIZE);
        (void)dot; // Suppress unused warning
    }
    end = clock();
    cpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;
    printf("Vector operations (%d iterations): %.4f seconds\n\n", VECTOR_ITERATIONS, cpu_time_used);
    
    // Benchmark 3: Trigonometry Functions
    printf("Running trigonometry functions benchmark...\n");
    double trig_result = 0.0;
    start = clock();
    for (i = 0; i < TRIG_ITERATIONS; i++) {
        double angle = (double)i / 100.0;
        trig_result += fast_sin(angle);
        trig_result += fast_cos(angle);
        trig_result += fast_tan(angle);
    }
    end = clock();
    cpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;
    printf("Trigonometry functions (%d iterations): %.4f seconds\n", TRIG_ITERATIONS, cpu_time_used);
    printf("Trig result checksum: %.6f\n\n", trig_result);
    
    // Benchmark 4: Statistics Functions
    printf("Running statistics functions benchmark...\n");
    double stats_result = 0.0;
    start = clock();
    for (i = 0; i < STATS_ITERATIONS; i++) {
        stats_result += compute_mean(vector_a, VECTOR_SIZE);
        stats_result += compute_variance(vector_a, VECTOR_SIZE);
        stats_result += compute_stddev(vector_a, VECTOR_SIZE);
        stats_result += find_min(vector_a, VECTOR_SIZE);
        stats_result += find_max(vector_a, VECTOR_SIZE);
    }
    end = clock();
    cpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;
    printf("Statistics functions (%d iterations): %.4f seconds\n", STATS_ITERATIONS, cpu_time_used);
    printf("Stats result checksum: %.6f\n\n", stats_result);
    
    // Clean up allocated memory
    free(matrix_a);
    free(matrix_b);
    free(matrix_result);
    free(vector_a);
    free(vector_b);
    free(vector_result);
    
    printf("All benchmarks completed successfully!\n");
    printf("BENCHMARK PASSED\n");
    
    return 0;
}
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

// Function prototypes from compute.c
void vector_add(double* a, double* b, double* result, int n);
void vector_multiply(double* a, double* b, double* result, int n);
void vector_fma(double* a, double* b, double* c, double* result, int n);
void vector_scale(double* a, double scalar, double* result, int n);

#define ARRAY_SIZE 50000
#define EPSILON 1e-9

int main() {
    printf("Vector Library Benchmark\n");
    printf("=========================\n");
    printf("Array size: %d elements\n\n", ARRAY_SIZE);
    
    // Allocate arrays
    double* a = (double*)malloc(ARRAY_SIZE * sizeof(double));
    double* b = (double*)malloc(ARRAY_SIZE * sizeof(double));
    double* c = (double*)malloc(ARRAY_SIZE * sizeof(double));
    double* result = (double*)malloc(ARRAY_SIZE * sizeof(double));
    
    if (!a || !b || !c || !result) {
        fprintf(stderr, "Memory allocation failed\n");
        return 1;
    }
    
    // Initialize arrays with test values
    for (int i = 0; i < ARRAY_SIZE; i++) {
        a[i] = (double)(i % 100) + 1.0;
        b[i] = (double)(i % 50) + 2.0;
        c[i] = (double)(i % 75) + 3.0;
        result[i] = 0.0;
    }
    
    clock_t start, end;
    double cpu_time;
    
    // Test 1: Vector Addition
    printf("Testing vector_add...\n");
    start = clock();
    vector_add(a, b, result, ARRAY_SIZE);
    end = clock();
    cpu_time = ((double)(end - start)) / CLOCKS_PER_SEC * 1000.0;
    
    // Validate result
    int errors = 0;
    for (int i = 0; i < 10 && i < ARRAY_SIZE; i++) {
        double expected = a[i] + b[i];
        if (fabs(result[i] - expected) > EPSILON) {
            errors++;
        }
    }
    printf("  Time: %.3f ms, Validation: %s\n", cpu_time, errors == 0 ? "PASSED" : "FAILED");
    
    // Test 2: Vector Multiplication
    printf("Testing vector_multiply...\n");
    start = clock();
    vector_multiply(a, b, result, ARRAY_SIZE);
    end = clock();
    cpu_time = ((double)(end - start)) / CLOCKS_PER_SEC * 1000.0;
    
    // Validate result
    errors = 0;
    for (int i = 0; i < 10 && i < ARRAY_SIZE; i++) {
        double expected = a[i] * b[i];
        if (fabs(result[i] - expected) > EPSILON) {
            errors++;
        }
    }
    printf("  Time: %.3f ms, Validation: %s\n", cpu_time, errors == 0 ? "PASSED" : "FAILED");
    
    // Test 3: Vector FMA (Fused Multiply-Add)
    printf("Testing vector_fma...\n");
    start = clock();
    vector_fma(a, b, c, result, ARRAY_SIZE);
    end = clock();
    cpu_time = ((double)(end - start)) / CLOCKS_PER_SEC * 1000.0;
    
    // Validate result
    errors = 0;
    for (int i = 0; i < 10 && i < ARRAY_SIZE; i++) {
        double expected = a[i] * b[i] + c[i];
        if (fabs(result[i] - expected) > EPSILON) {
            errors++;
        }
    }
    printf("  Time: %.3f ms, Validation: %s\n", cpu_time, errors == 0 ? "PASSED" : "FAILED");
    
    // Test 4: Vector Scaling
    printf("Testing vector_scale...\n");
    double scalar = 2.5;
    start = clock();
    vector_scale(a, scalar, result, ARRAY_SIZE);
    end = clock();
    cpu_time = ((double)(end - start)) / CLOCKS_PER_SEC * 1000.0;
    
    // Validate result
    errors = 0;
    for (int i = 0; i < 10 && i < ARRAY_SIZE; i++) {
        double expected = a[i] * scalar;
        if (fabs(result[i] - expected) > EPSILON) {
            errors++;
        }
    }
    printf("  Time: %.3f ms, Validation: %s\n", cpu_time, errors == 0 ? "PASSED" : "FAILED");
    
    // Cleanup
    free(a);
    free(b);
    free(c);
    free(result);
    
    printf("\nBenchmark completed successfully!\n");
    return 0;
}
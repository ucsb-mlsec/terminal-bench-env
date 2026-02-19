#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <string.h>

#define TEST_SIZE 10000000
#define ITERATIONS 100
#define EPSILON 1e-6

extern double vector_sum(const double* array, size_t length);

int compare_doubles(double a, double b, double epsilon) {
    return fabs(a - b) < epsilon;
}

double simple_sum(const double* array, size_t length) {
    double sum = 0.0;
    for (size_t i = 0; i < length; i++) {
        sum += array[i];
    }
    return sum;
}

int main() {
    double* array = (double*)malloc(TEST_SIZE * sizeof(double));
    if (!array) {
        fprintf(stderr, "Memory allocation failed\n");
        return 1;
    }

    printf("Initializing test array with %d elements...\n", TEST_SIZE);
    for (size_t i = 0; i < TEST_SIZE; i++) {
        array[i] = 1.0 + (i % 100) * 0.01;
    }

    printf("Calculating expected sum...\n");
    double expected_sum = 0.0;
    for (size_t i = 0; i < TEST_SIZE; i++) {
        expected_sum += array[i];
    }
    printf("Expected sum: %.6f\n", expected_sum);

    printf("Warming up...\n");
    double warmup_result = vector_sum(array, TEST_SIZE);

    printf("Running performance benchmark (%d iterations)...\n", ITERATIONS);
    clock_t start = clock();
    double result = 0.0;
    for (int i = 0; i < ITERATIONS; i++) {
        result = vector_sum(array, TEST_SIZE);
    }
    clock_t end = clock();

    double total_time_ms = ((double)(end - start) / CLOCKS_PER_SEC) * 1000.0;
    double avg_time_ms = total_time_ms / ITERATIONS;

    printf("\n=== PERFORMANCE RESULTS ===\n");
    printf("Total time for %d iterations: %.2f ms\n", ITERATIONS, total_time_ms);
    printf("Average time per iteration: %.4f ms\n", avg_time_ms);
    printf("BASELINE TIME: %.4f ms\n", avg_time_ms);

    printf("\n=== CORRECTNESS VERIFICATION ===\n");
    printf("Expected sum: %.6f\n", expected_sum);
    printf("Computed sum: %.6f\n", result);
    printf("Difference: %.10f\n", fabs(result - expected_sum));
    
    int passed = compare_doubles(result, expected_sum, EPSILON);
    if (passed) {
        printf("CORRECTNESS: PASS\n");
    } else {
        printf("CORRECTNESS: FAIL\n");
        printf("Error exceeds tolerance (epsilon = %e)\n", EPSILON);
    }

    free(array);

    return passed ? 0 : 1;
}
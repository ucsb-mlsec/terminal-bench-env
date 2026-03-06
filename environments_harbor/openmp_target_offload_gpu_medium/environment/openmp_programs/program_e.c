#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define N 10000

int main() {
    int i;
    double *a, *b;
    double sum = 0.0;
    
    // Allocate memory on host
    a = (double*)malloc(N * sizeof(double));
    b = (double*)malloc(N * sizeof(double));
    
    if (a == NULL || b == NULL) {
        fprintf(stderr, "Memory allocation failed\n");
        return 1;
    }
    
    // Initialize arrays
    for (i = 0; i < N; i++) {
        a[i] = (double)(i + 1);
        b[i] = (double)(i * 2);
    }
    
    // Compute dot product using target offload
    // ERROR: Missing reduction clause for sum variable
    // This will cause a race condition as multiple threads
    // try to update sum simultaneously without proper synchronization
    #pragma omp target map(to: a[0:N], b[0:N]) map(tofrom: sum)
    #pragma omp teams distribute parallel for
    for (i = 0; i < N; i++) {
        sum += a[i] * b[i];
    }
    
    printf("Dot product result: %f\n", sum);
    
    // Verify result on CPU
    double expected_sum = 0.0;
    for (i = 0; i < N; i++) {
        expected_sum += a[i] * b[i];
    }
    
    printf("Expected result: %f\n", expected_sum);
    printf("Difference: %f\n", sum - expected_sum);
    
    if (sum != expected_sum) {
        printf("WARNING: Results do not match - race condition detected!\n");
    }
    
    // Find maximum value using another flawed reduction
    // ERROR: Using wrong reduction operator (+ instead of max)
    double max_val = a[0];
    #pragma omp target map(to: a[0:N]) map(tofrom: max_val)
    #pragma omp teams distribute parallel for reduction(+:max_val)
    for (i = 0; i < N; i++) {
        if (a[i] > max_val) {
            max_val = a[i];
        }
    }
    
    printf("Maximum value found: %f\n", max_val);
    printf("Expected maximum: %f\n", (double)N);
    
    // Clean up
    free(a);
    free(b);
    
    return 0;
}
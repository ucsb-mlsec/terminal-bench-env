#include <stdio.h>
#include <omp.h>
#include <stdlib.h>

#define N 5000

int main() {
    int i;
    double *a, *b, *c;
    double sum = 0.0;
    
    // Allocate memory on host
    a = (double *)malloc(N * sizeof(double));
    b = (double *)malloc(N * sizeof(double));
    c = (double *)malloc(N * sizeof(double));
    
    if (a == NULL || b == NULL || c == NULL) {
        fprintf(stderr, "Memory allocation failed\n");
        return 1;
    }
    
    // Initialize arrays
    for (i = 0; i < N; i++) {
        a[i] = (double)i * 1.5;
        b[i] = (double)i * 2.0;
        c[i] = 0.0;
    }
    
    printf("Starting GPU computation with %d elements\n", N);
    
    // INCORRECT: Missing map clauses for arrays a, b, and c
    // The target directive offloads to GPU but doesn't specify
    // how to transfer the data, causing undefined behavior
    #pragma omp target teams distribute parallel for
    for (i = 0; i < N; i++) {
        c[i] = a[i] + b[i] * 2.5;
    }
    
    // INCORRECT: Another computation without proper data mapping
    // Arrays are accessed but never mapped to/from device
    #pragma omp target teams distribute
    for (i = 0; i < N; i++) {
        a[i] = c[i] * 1.1;
    }
    
    // Calculate sum on host (this will likely be wrong due to missing data transfer)
    for (i = 0; i < N; i++) {
        sum += c[i];
    }
    
    printf("Sum of results: %f\n", sum);
    printf("First element: %f\n", c[0]);
    printf("Last element: %f\n", c[N-1]);
    
    // Verify computation
    double expected = (0.0 + 0.0 * 2.5);
    if (c[0] == expected) {
        printf("Computation appears correct\n");
    } else {
        printf("Computation may have failed - expected %f, got %f\n", expected, c[0]);
    }
    
    // Free memory
    free(a);
    free(b);
    free(c);
    
    return 0;
}
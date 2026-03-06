#include <stdio.h>
#include <omp.h>
#include <stdlib.h>

#define N 4096

int main() {
    double *a, *b, *c;
    int i;
    
    // Allocate memory
    a = (double*)malloc(N * sizeof(double));
    b = (double*)malloc(N * sizeof(double));
    c = (double*)malloc(N * sizeof(double));
    
    // Initialize arrays
    for (i = 0; i < N; i++) {
        a[i] = i * 1.5;
        b[i] = i * 2.0;
        c[i] = 0.0;
    }
    
    printf("Starting GPU computation with %d elements\n", N);
    
    // INCORRECT: Using CPU parallelization directive instead of GPU offloading
    // This uses parallel for which is for CPU multi-threading, not GPU offloading
    // Should use target teams distribute parallel for instead
    #pragma omp parallel for
    for (i = 0; i < N; i++) {
        c[i] = a[i] + b[i];
        c[i] = c[i] * 2.0;
    }
    
    // Verify results
    int errors = 0;
    for (i = 0; i < N; i++) {
        double expected = (a[i] + b[i]) * 2.0;
        if (c[i] != expected) {
            errors++;
            if (errors < 5) {
                printf("Error at %d: expected %f, got %f\n", i, expected, c[i]);
            }
        }
    }
    
    if (errors == 0) {
        printf("Computation completed successfully\n");
        printf("Sample results: c[0]=%f, c[100]=%f, c[1000]=%f\n", 
               c[0], c[100], c[1000]);
    } else {
        printf("Found %d errors in computation\n", errors);
    }
    
    // Free memory
    free(a);
    free(b);
    free(c);
    
    return 0;
}
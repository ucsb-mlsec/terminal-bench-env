#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define N 8192

int main() {
    double *a, *b, *c;
    int i;
    
    // Allocate memory for arrays
    a = (double *)malloc(N * sizeof(double));
    b = (double *)malloc(N * sizeof(double));
    c = (double *)malloc(N * sizeof(double));
    
    // Initialize arrays
    for (i = 0; i < N; i++) {
        a[i] = (double)i * 1.5;
        b[i] = (double)i * 2.5;
        c[i] = 0.0;
    }
    
    printf("Starting GPU computation with %d elements\n", N);
    
    // INCORRECT: Missing #pragma omp target directive
    // This attempts to use teams distribute but without target,
    // it will not execute on the GPU device
    #pragma omp teams distribute parallel for
    for (i = 0; i < N; i++) {
        c[i] = a[i] * b[i] + a[i];
    }
    
    // Verify results
    int errors = 0;
    for (i = 0; i < N; i++) {
        double expected = a[i] * b[i] + a[i];
        if (c[i] != expected) {
            errors++;
            if (errors < 5) {
                printf("Error at index %d: expected %f, got %f\n", 
                       i, expected, c[i]);
            }
        }
    }
    
    if (errors == 0) {
        printf("Computation completed successfully!\n");
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
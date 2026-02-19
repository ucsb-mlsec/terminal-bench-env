#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define N 100000

int main() {
    int i;
    double *a, *b, *c;
    
    // Allocate memory for arrays
    a = (double*)malloc(N * sizeof(double));
    b = (double*)malloc(N * sizeof(double));
    c = (double*)malloc(N * sizeof(double));
    
    if (a == NULL || b == NULL || c == NULL) {
        printf("Memory allocation failed\n");
        return 1;
    }
    
    // Initialize input arrays
    for (i = 0; i < N; i++) {
        a[i] = (double)i;
        b[i] = (double)(i * 2);
        c[i] = 0.0;
    }
    
    printf("Starting vector addition on GPU...\n");
    printf("Vector size: %d elements\n", N);
    
    // Offload computation to GPU with proper data mapping
    #pragma omp target teams distribute parallel for \
        map(to: a[0:N], b[0:N]) \
        map(from: c[0:N])
    for (i = 0; i < N; i++) {
        c[i] = a[i] + b[i];
    }
    
    // Verify results on a subset
    int errors = 0;
    for (i = 0; i < 10; i++) {
        double expected = a[i] + b[i];
        if (c[i] != expected) {
            errors++;
            printf("Error at index %d: expected %f, got %f\n", 
                   i, expected, c[i]);
        }
    }
    
    if (errors == 0) {
        printf("Vector addition completed successfully!\n");
        printf("Sample results:\n");
        for (i = 0; i < 5; i++) {
            printf("c[%d] = a[%d] + b[%d] = %f + %f = %f\n", 
                   i, i, i, a[i], b[i], c[i]);
        }
    } else {
        printf("Vector addition had %d errors\n", errors);
    }
    
    // Check last few elements as well
    printf("Last element: c[%d] = %f (expected %f)\n", 
           N-1, c[N-1], a[N-1] + b[N-1]);
    
    // Free allocated memory
    free(a);
    free(b);
    free(c);
    
    return 0;
}
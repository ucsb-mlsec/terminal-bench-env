#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define N 10000

int main() {
    double *a, *b, *c;
    int i;
    
    // Allocate memory
    a = (double*)malloc(N * sizeof(double));
    b = (double*)malloc(N * sizeof(double));
    c = (double*)malloc(N * sizeof(double));
    
    // Initialize arrays
    for(i = 0; i < N; i++) {
        a[i] = i * 1.0;
        b[i] = i * 2.0;
        c[i] = 0.0;
    }
    
    // GPU offloading with proper target directive
    #pragma omp target map(to: a[0:N], b[0:N]) map(from: c[0:N])
    #pragma omp teams distribute parallel for
    for(i = 0; i < N; i++) {
        c[i] = a[i] + b[i];
    }
    
    // Verify results
    int errors = 0;
    for(i = 0; i < N; i++) {
        if(c[i] != a[i] + b[i]) {
            errors++;
        }
    }
    
    if(errors == 0) {
        printf("Vector addition completed successfully on GPU\n");
        printf("Sample results: c[0]=%.2f, c[100]=%.2f, c[N-1]=%.2f\n", 
               c[0], c[100], c[N-1]);
    } else {
        printf("Errors detected: %d\n", errors);
    }
    
    // Free memory
    free(a);
    free(b);
    free(c);
    
    return 0;
}
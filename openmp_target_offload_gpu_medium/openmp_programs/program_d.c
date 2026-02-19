#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define N 10000

int main() {
    int i;
    double *a, *b, *c;
    double sum = 0.0;
    double max_val = 0.0;
    
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
        a[i] = (double)i * 0.5;
        b[i] = (double)i * 0.3;
        c[i] = 0.0;
    }
    
    // Offload vector addition to GPU
    // Map input arrays to device, output array from device
    #pragma omp target map(to: a[0:N], b[0:N]) map(from: c[0:N])
    #pragma omp parallel for private(i)
    for (i = 0; i < N; i++) {
        c[i] = a[i] + b[i];
    }
    
    // Offload reduction operation to GPU
    // Map array to device, reduce sum back to host
    #pragma omp target map(to: c[0:N]) map(tofrom: sum)
    #pragma omp parallel for private(i) reduction(+:sum)
    for (i = 0; i < N; i++) {
        sum += c[i];
    }
    
    // Offload array scaling operation
    #pragma omp target map(tofrom: c[0:N])
    #pragma omp parallel for private(i)
    for (i = 0; i < N; i++) {
        c[i] = c[i] * 2.0;
    }
    
    // Offload max reduction to GPU
    #pragma omp target map(to: c[0:N]) map(tofrom: max_val)
    #pragma omp parallel for private(i) reduction(max:max_val)
    for (i = 0; i < N; i++) {
        if (c[i] > max_val) {
            max_val = c[i];
        }
    }
    
    // Print results
    printf("Sum of elements: %f\n", sum);
    printf("Maximum value: %f\n", max_val);
    printf("First 5 elements of c: ");
    for (i = 0; i < 5; i++) {
        printf("%f ", c[i]);
    }
    printf("\n");
    
    // Free memory
    free(a);
    free(b);
    free(c);
    
    return 0;
}
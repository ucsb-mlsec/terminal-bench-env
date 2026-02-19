#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define N 10000

int main() {
    double *x, *y, *result;
    double dot_product = 0.0;
    int i;

    // Allocate memory on host
    x = (double *)malloc(N * sizeof(double));
    y = (double *)malloc(N * sizeof(double));
    result = (double *)malloc(N * sizeof(double));

    if (x == NULL || y == NULL || result == NULL) {
        printf("Memory allocation failed\n");
        return 1;
    }

    // Initialize arrays
    for (i = 0; i < N; i++) {
        x[i] = (double)(i + 1);
        y[i] = (double)(2 * i + 1);
        result[i] = 0.0;
    }

    // Perform element-wise multiplication on GPU
    #pragma omp target teams distribute parallel for map(to: x[0:N], y[0:N]) map(from: result[0:N])
    for (i = 0; i < N; i++) {
        result[i] = x[i] * y[i];
    }

    // Compute dot product on host from results
    for (i = 0; i < N; i++) {
        dot_product += result[i];
    }

    // Verify some results
    printf("Element-wise multiplication completed on GPU\n");
    printf("First element: x[0] * y[0] = %.2f * %.2f = %.2f\n", 
           x[0], y[0], result[0]);
    printf("Last element: x[%d] * y[%d] = %.2f * %.2f = %.2f\n", 
           N-1, N-1, x[N-1], y[N-1], result[N-1]);
    printf("Dot product sum: %.2f\n", dot_product);

    // Expected first result: 1.0 * 1.0 = 1.0
    // Expected last result: 10000.0 * 19999.0 = 199990000.0
    if (result[0] == 1.0 && result[N-1] == 199990000.0) {
        printf("Verification: PASSED\n");
    } else {
        printf("Verification: FAILED\n");
    }

    // Free memory
    free(x);
    free(y);
    free(result);

    return 0;
}
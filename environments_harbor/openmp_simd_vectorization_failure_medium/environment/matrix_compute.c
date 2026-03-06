#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define SIZE 2048
#define TILE_SIZE 64

// Computation kernel that performs matrix-style operations
void compute_kernel(float *input_a, float *input_b, float *output, int n) {
    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            float sum = 0.0f;
            // Pointer aliasing issue: compiler cannot prove that output doesn't
            // overlap with input arrays, preventing vectorization
            float *pa = input_a + i * n;
            float *pb = input_b + j;
            float *pout = output + i * n + j;
            
            for (int k = 0; k < n; k++) {
                sum += pa[k] * pb[k * n];
            }
            *pout = sum;
        }
    }
}

// Secondary kernel with element-wise operations
void apply_transform(float *data, float *result, int n) {
    #pragma omp parallel for
    for (int i = 0; i < n * n; i++) {
        // Pointer arithmetic prevents vectorization analysis
        float *d = data + i;
        float *r = result + i;
        *r = (*d) * 1.5f + 2.0f;
    }
}

int main() {
    int n = SIZE;
    size_t array_size = n * n * sizeof(float);
    
    // Allocate arrays
    float *matrix_a = (float*)malloc(array_size);
    float *matrix_b = (float*)malloc(array_size);
    float *matrix_c = (float*)malloc(array_size);
    float *matrix_result = (float*)malloc(array_size);
    
    if (!matrix_a || !matrix_b || !matrix_c || !matrix_result) {
        fprintf(stderr, "Memory allocation failed\n");
        return 1;
    }
    
    // Initialize input matrices
    for (int i = 0; i < n * n; i++) {
        matrix_a[i] = (float)(i % 100) / 100.0f;
        matrix_b[i] = (float)((i + 1) % 100) / 50.0f;
    }
    
    // Perform computation
    double start_time = omp_get_wtime();
    
    compute_kernel(matrix_a, matrix_b, matrix_c, n);
    apply_transform(matrix_c, matrix_result, n);
    
    double end_time = omp_get_wtime();
    
    // Compute checksum and sample values for verification
    double checksum = 0.0;
    for (int i = 0; i < n * n; i++) {
        checksum += matrix_result[i];
    }
    
    // Print results
    printf("Computation completed\n");
    printf("Matrix size: %d x %d\n", n, n);
    printf("Checksum: %.6f\n", checksum);
    printf("Time: %.4f seconds\n", end_time - start_time);
    printf("Sample values:\n");
    printf("  [0][0] = %.6f\n", matrix_result[0]);
    printf("  [10][10] = %.6f\n", matrix_result[10 * n + 10]);
    printf("  [100][100] = %.6f\n", matrix_result[100 * n + 100]);
    printf("  [500][500] = %.6f\n", matrix_result[500 * n + 500]);
    
    // Cleanup
    free(matrix_a);
    free(matrix_b);
    free(matrix_c);
    free(matrix_result);
    
    return 0;
}
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define N 1000000
#define STRIDE 3

// Loop 1: Pointer aliasing problem - no restrict qualifiers
void compute1(float *a, float *b, float *c, int n) {
    #pragma omp parallel for
    for(int i = 0; i < n; i++) {
        c[i] = a[i] + b[i];
    }
}

// Loop 2: Non-contiguous memory access with stride
void compute2(float *data, float *result, int n, int stride) {
    #pragma omp parallel for
    for(int i = 0; i < n; i++) {
        result[i] = data[i * stride] * 2.0f;
    }
}

// Loop 3: Loop-carried dependency
void compute3(float *a, int n) {
    #pragma omp parallel for
    for(int i = 1; i < n; i++) {
        a[i] = a[i] + a[i-1];
    }
}

int main() {
    float *array1, *array2, *array3, *result1, *result2;
    float *strided_data, *strided_result;
    
    // Allocate memory for Loop 1
    array1 = (float*)malloc(N * sizeof(float));
    array2 = (float*)malloc(N * sizeof(float));
    result1 = (float*)malloc(N * sizeof(float));
    
    if(array1 == NULL || array2 == NULL || result1 == NULL) {
        fprintf(stderr, "Memory allocation failed for Loop 1\n");
        return 1;
    }
    
    // Allocate memory for Loop 2 (strided access)
    strided_data = (float*)malloc(N * STRIDE * sizeof(float));
    strided_result = (float*)malloc(N * sizeof(float));
    
    if(strided_data == NULL || strided_result == NULL) {
        fprintf(stderr, "Memory allocation failed for Loop 2\n");
        free(array1);
        free(array2);
        free(result1);
        return 1;
    }
    
    // Allocate memory for Loop 3
    array3 = (float*)malloc(N * sizeof(float));
    
    if(array3 == NULL) {
        fprintf(stderr, "Memory allocation failed for Loop 3\n");
        free(array1);
        free(array2);
        free(result1);
        free(strided_data);
        free(strided_result);
        return 1;
    }
    
    // Initialize arrays
    for(int i = 0; i < N; i++) {
        array1[i] = (float)i;
        array2[i] = (float)(i * 2);
        array3[i] = 1.0f;
    }
    
    for(int i = 0; i < N * STRIDE; i++) {
        strided_data[i] = (float)i;
    }
    
    printf("Starting computations...\n");
    
    // Execute Loop 1: Element-wise addition with pointer aliasing
    compute1(array1, array2, result1, N);
    printf("Loop 1 completed: result1[100] = %f\n", result1[100]);
    
    // Execute Loop 2: Non-contiguous memory access
    compute2(strided_data, strided_result, N, STRIDE);
    printf("Loop 2 completed: strided_result[100] = %f\n", strided_result[100]);
    
    // Execute Loop 3: Loop-carried dependency
    compute3(array3, N);
    printf("Loop 3 completed: array3[10] = %f\n", array3[10]);
    
    // Verification output
    printf("\nVerification:\n");
    printf("Loop 1 check (should be 300): %f\n", result1[100]);
    printf("Loop 2 check (should be 600): %f\n", strided_result[100]);
    printf("Loop 3 check (should be 11): %f\n", array3[10]);
    
    // Cleanup
    free(array1);
    free(array2);
    free(array3);
    free(result1);
    free(strided_data);
    free(strided_result);
    
    printf("\nComputations completed successfully.\n");
    
    return 0;
}
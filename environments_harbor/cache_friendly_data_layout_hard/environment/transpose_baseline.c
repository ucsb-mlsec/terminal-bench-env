#include <stdio.h>
#include <stdlib.h>

#define N 8192

int main() {
    double *input = NULL;
    double *output = NULL;
    FILE *fp = NULL;
    
    // Allocate memory for input and output matrices
    input = (double*)malloc(N * N * sizeof(double));
    if (input == NULL) {
        fprintf(stderr, "Error: Failed to allocate memory for input matrix\n");
        return 1;
    }
    
    output = (double*)malloc(N * N * sizeof(double));
    if (output == NULL) {
        fprintf(stderr, "Error: Failed to allocate memory for output matrix\n");
        free(input);
        return 1;
    }
    
    // Read input matrix from binary file
    fp = fopen("/workspace/test_matrix.bin", "rb");
    if (fp == NULL) {
        fprintf(stderr, "Error: Failed to open input file\n");
        free(input);
        free(output);
        return 1;
    }
    
    size_t elements_read = fread(input, sizeof(double), N * N, fp);
    if (elements_read != N * N) {
        fprintf(stderr, "Error: Failed to read complete matrix. Read %zu elements, expected %d\n", 
                elements_read, N * N);
        fclose(fp);
        free(input);
        free(output);
        return 1;
    }
    fclose(fp);
    
    // Naive transpose - causes cache thrashing
    // This implementation has terrible cache locality because it accesses
    // output memory in column-major order while it's stored in row-major order
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            output[j * N + i] = input[i * N + j];
        }
    }
    
    // Write transposed matrix to binary file
    fp = fopen("/workspace/output_transposed.bin", "wb");
    if (fp == NULL) {
        fprintf(stderr, "Error: Failed to open output file\n");
        free(input);
        free(output);
        return 1;
    }
    
    size_t elements_written = fwrite(output, sizeof(double), N * N, fp);
    if (elements_written != N * N) {
        fprintf(stderr, "Error: Failed to write complete matrix. Wrote %zu elements, expected %d\n",
                elements_written, N * N);
        fclose(fp);
        free(input);
        free(output);
        return 1;
    }
    fclose(fp);
    
    // Clean up
    free(input);
    free(output);
    
    return 0;
}
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define ARRAY_SIZE 1024
#define BLOCK_SIZE 256

/* Traditional CUDA memory allocation example */

int initialize_device_memory(float **d_input, float **d_output, size_t size) {
    cudaError_t err;
    
    /* Allocate input buffer using traditional cudaMalloc */
    err = cudaMalloc((void**)d_input, size * sizeof(float));
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate device input: %s\n", 
                cudaGetErrorString(err));
        return -1;
    }
    
    /* Allocate output buffer using traditional cudaMalloc */
    err = cudaMalloc((void**)d_output, size * sizeof(float));
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate device output: %s\n",
                cudaGetErrorString(err));
        cudaFree(*d_input);
        return -1;
    }
    
    return 0;
}

void cleanup_device_memory(float *d_input, float *d_output) {
    /* Free allocated device memory */
    if (d_input != NULL) {
        cudaFree(d_input);
    }
    if (d_output != NULL) {
        cudaFree(d_output);
    }
}

int main() {
    float *d_input = NULL;
    float *d_output = NULL;
    
    if (initialize_device_memory(&d_input, &d_output, ARRAY_SIZE) != 0) {
        return EXIT_FAILURE;
    }
    
    /* Perform operations here */
    
    cleanup_device_memory(d_input, d_output);
    return EXIT_SUCCESS;
}
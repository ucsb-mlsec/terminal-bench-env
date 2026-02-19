#include <cuda_runtime.h>
#include <stdio.h>

// Simple vector addition kernel
__global__ void vectorAdd(const float *a, const float *b, float *c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

// Host function that performs vector addition using traditional allocation
int performVectorAddition(const float *h_a, const float *h_b, float *h_c, int n) {
    float *d_a = NULL;
    float *d_b = NULL;
    float *d_c = NULL;
    size_t size = n * sizeof(float);
    
    // Traditional CUDA memory allocation
    cudaError_t err = cudaMalloc((void**)&d_a, size);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate device memory for d_a\n");
        return -1;
    }
    
    err = cudaMalloc((void**)&d_b, size);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate device memory for d_b\n");
        cudaFree(d_a);
        return -1;
    }
    
    err = cudaMalloc((void**)&d_c, size);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate device memory for d_c\n");
        cudaFree(d_a);
        cudaFree(d_b);
        return -1;
    }
    
    // Copy data from host to device
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);
    
    // Launch kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, n);
    
    // Copy result back to host
    cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);
    
    // Traditional cleanup with cudaFree
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    
    return 0;
}
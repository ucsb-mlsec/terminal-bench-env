#include <cuda_runtime.h>
#include <stdio.h>

__global__ void vectorAdd(float* a, float* b, float* c, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < size) {
        c[idx] = a[idx] + b[idx];
    }
}

int launchKernel(float* host_a, float* host_b, float* host_c, int size) {
    float *dev_a, *dev_b, *dev_c;
    size_t bytes = size * sizeof(float);
    
    // Allocate device memory
    cudaError_t err = cudaMalloc((void**)&dev_a, bytes);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate device memory for a: %s\n", cudaGetErrorString(err));
        return -1;
    }
    
    err = cudaMalloc((void**)&dev_b, bytes);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate device memory for b: %s\n", cudaGetErrorString(err));
        cudaFree(dev_a);
        return -1;
    }
    
    err = cudaMalloc((void**)&dev_c, bytes);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate device memory for c: %s\n", cudaGetErrorString(err));
        cudaFree(dev_a);
        cudaFree(dev_b);
        return -1;
    }
    
    // Copy data to device
    cudaMemcpy(dev_a, host_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, host_b, bytes, cudaMemcpyHostToDevice);
    
    // Launch kernel
    int blockSize = 256;
    int gridSize = (size + blockSize - 1) / blockSize;
    vectorAdd<<<gridSize, blockSize>>>(dev_a, dev_b, dev_c, size);
    
    // Check for kernel launch errors
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err));
        cudaFree(dev_a);
        cudaFree(dev_b);
        cudaFree(dev_c);
        return -1;
    }
    
    // Copy results back to host
    cudaMemcpy(host_c, dev_c, bytes, cudaMemcpyDeviceToHost);
    
    // Free device memory
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);
    
    return 0;
}
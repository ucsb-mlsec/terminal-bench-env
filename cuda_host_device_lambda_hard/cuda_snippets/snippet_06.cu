// Invalid CUDA C++ snippet demonstrating incorrect lambda usage
// Error: Attempting to use a host-only lambda (without __device__ specifier) within a device kernel

#include <cuda_runtime.h>

__global__ void processKernel(int* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // ERROR: This lambda has no __device__ specifier, making it host-only
    // It cannot be called from device code
    auto transform = [](int x) {
        return x * 2 + 1;
    };
    
    if (idx < n) {
        data[idx] = transform(data[idx]);  // This will fail compilation
    }
}

void launchKernel(int* d_data, int n) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    
    processKernel<<<blocksPerGrid, threadsPerBlock>>>(d_data, n);
}
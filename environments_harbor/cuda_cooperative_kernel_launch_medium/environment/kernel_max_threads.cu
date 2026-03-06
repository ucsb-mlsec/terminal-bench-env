#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <stdio.h>

// Cooperative kernel that attempts to use grid-wide synchronization
__global__ void cooperativeKernel(int *data, int size) {
    // Use cooperative groups for grid-wide synchronization
    cooperative_groups::grid_group grid = cooperative_groups::this_grid();
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < size) {
        // Perform some computation
        data[tid] = data[tid] * 2;
    }
    
    // Synchronize all blocks in the grid
    grid.sync();
    
    if (tid < size) {
        // Further computation after synchronization
        data[tid] = data[tid] + 1;
    }
}

int main() {
    int size = 153600;  // 100 blocks * 1536 threads
    int *d_data;
    
    // Allocate device memory
    cudaMalloc(&d_data, size * sizeof(int));
    
    // Configure cooperative kernel launch parameters
    // Grid: 100 blocks
    // Block: 1536 threads per block (EXCEEDS max threads per block limit!)
    dim3 gridDim(100, 1, 1);
    dim3 blockDim(1536, 1, 1);
    
    // Set up launch parameters for cooperative kernel
    void *kernelArgs[] = {&d_data, &size};
    
    // Query device properties to check cooperative launch support
    int device = 0;
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, device);
    
    printf("Launching cooperative kernel with:\n");
    printf("  Grid dimensions: (%d, %d, %d)\n", gridDim.x, gridDim.y, gridDim.z);
    printf("  Block dimensions: (%d, %d, %d)\n", blockDim.x, blockDim.y, blockDim.z);
    printf("  Total threads per block: %d\n", blockDim.x * blockDim.y * blockDim.z);
    printf("  Total blocks: %d\n", gridDim.x * gridDim.y * gridDim.z);
    
    // Launch cooperative kernel
    // Note: This configuration is INVALID because 1536 threads per block
    // exceeds the maximum threads per block limit (typically 1024)
    cudaLaunchCooperativeKernel((void*)cooperativeKernel, gridDim, blockDim, kernelArgs);
    
    // Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Cooperative kernel launch failed: %s\n", cudaGetErrorString(err));
    }
    
    // Synchronize and cleanup
    cudaDeviceSynchronize();
    cudaFree(d_data);
    
    return 0;
}
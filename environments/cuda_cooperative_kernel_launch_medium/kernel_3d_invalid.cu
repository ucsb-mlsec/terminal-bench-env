#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <stdio.h>

namespace cg = cooperative_groups;

// Cooperative kernel that uses 3D grid synchronization
// This kernel attempts to use a 3D block configuration
__global__ void cooperativeKernel3D(int* data, int size) {
    // Get cooperative grid group handle
    cg::grid_group grid = cg::this_grid();
    
    // Calculate 3D thread index
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;
    
    // Linear index from 3D coordinates
    int idx = z * (gridDim.x * blockDim.x * gridDim.y * blockDim.y) +
              y * (gridDim.x * blockDim.x) + x;
    
    if (idx < size) {
        // First phase: all threads write initial values
        data[idx] = idx * 2;
        
        // Grid-wide synchronization - requires cooperative launch
        grid.sync();
        
        // Second phase: all threads can safely read values written by others
        if (idx > 0 && idx < size - 1) {
            data[idx] = (data[idx - 1] + data[idx + 1]) / 2;
        }
    }
}

int main() {
    const int dataSize = 100000;
    int* d_data;
    
    // Allocate device memory
    cudaMalloc(&d_data, dataSize * sizeof(int));
    
    // Configure 3D launch parameters
    // WARNING: This configuration has multiple issues for cooperative launch:
    // 1. Block Z dimension is 128, which exceeds maximum of 64
    // 2. Total threads per block: 8 * 8 * 128 = 8192, which exceeds maximum of 1024
    // 3. This will cause cooperative launch to fail
    dim3 gridDim(100, 1, 1);      // 100 blocks in X dimension
    dim3 blockDim(8, 8, 128);      // 8x8x128 = 8192 threads per block (INVALID!)
    
    // Calculate total blocks for cooperative launch
    int numBlocks = gridDim.x * gridDim.y * gridDim.z;
    
    printf("Launching cooperative kernel with 3D block configuration:\n");
    printf("Grid dimensions: (%d, %d, %d)\n", gridDim.x, gridDim.y, gridDim.z);
    printf("Block dimensions: (%d, %d, %d)\n", blockDim.x, blockDim.y, blockDim.z);
    printf("Total threads per block: %d\n", blockDim.x * blockDim.y * blockDim.z);
    printf("Total blocks: %d\n", numBlocks);
    
    // Set up kernel launch parameters for cooperative kernel
    void* kernelArgs[] = { &d_data, (void*)&dataSize };
    
    // Attempt to launch cooperative kernel with 3D block configuration
    // This will fail due to invalid block dimensions
    cudaError_t err = cudaLaunchCooperativeKernel(
        (void*)cooperativeKernel3D,
        gridDim,
        blockDim,
        kernelArgs,
        0,  // shared memory
        0   // stream
    );
    
    if (err != cudaSuccess) {
        printf("Cooperative kernel launch failed: %s\n", cudaGetErrorString(err));
        printf("Reason: Block Z dimension (%d) exceeds max (64), and total threads (%d) exceeds max (1024)\n",
               blockDim.z, blockDim.x * blockDim.y * blockDim.z);
    }
    
    // Cleanup
    cudaFree(d_data);
    
    return 0;
}
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <stdio.h>

namespace cg = cooperative_groups;

// Cooperative kernel using 2D grid and block configuration
// This kernel demonstrates grid-wide synchronization using cooperative groups
__global__ void cooperativeKernel2D(float* data, int width, int height) {
    // Get cooperative grid group for grid-wide synchronization
    cg::grid_group grid = cg::this_grid();
    
    // Calculate 2D thread index
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = y * width + x;
    
    // Perform some work
    if (x < width && y < height) {
        data[idx] = data[idx] * 2.0f;
    }
    
    // Grid-wide synchronization - this is what makes cooperative launch necessary
    grid.sync();
    
    // Second phase of computation after synchronization
    if (x < width && y < height) {
        data[idx] = data[idx] + 1.0f;
    }
}

int main() {
    // Configuration parameters
    const int width = 2048;
    const int height = 1024;
    const int dataSize = width * height;
    
    // Allocate device memory
    float* d_data;
    cudaMalloc(&d_data, dataSize * sizeof(float));
    
    // 2D Grid configuration for cooperative launch
    // Grid: 128 x 64 x 1 = 8192 total blocks (VALID: < 32768 max cooperative blocks)
    // Block: 16 x 16 x 1 = 256 threads per block (VALID: < 1024 max threads per block)
    dim3 gridDim(128, 64, 1);   // 2D grid covering the data space
    dim3 blockDim(16, 16, 1);   // 2D block with 256 threads
    
    // Calculate total blocks: 128 * 64 = 8192 blocks
    // This is well within the cooperative launch limit of 32768 blocks
    
    // Setup kernel arguments
    void* kernelArgs[] = { &d_data, (void*)&width, (void*)&height };
    
    // Query device properties to ensure cooperative launch is supported
    int device = 0;
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, device);
    
    // Verify cooperative launch support
    if (deviceProp.cooperativeLaunch) {
        printf("Launching cooperative kernel with 2D configuration:\n");
        printf("Grid dimensions: %d x %d x %d = %d blocks\n", 
               gridDim.x, gridDim.y, gridDim.z, gridDim.x * gridDim.y * gridDim.z);
        printf("Block dimensions: %d x %d x %d = %d threads\n", 
               blockDim.x, blockDim.y, blockDim.z, blockDim.x * blockDim.y * blockDim.z);
        
        // Launch cooperative kernel
        cudaLaunchCooperativeKernel(
            (void*)cooperativeKernel2D,
            gridDim,
            blockDim,
            kernelArgs,
            0,  // shared memory
            0   // stream
        );
        
        cudaDeviceSynchronize();
        printf("Cooperative kernel completed successfully\n");
    } else {
        printf("Cooperative launch not supported on this device\n");
    }
    
    // Cleanup
    cudaFree(d_data);
    
    return 0;
}
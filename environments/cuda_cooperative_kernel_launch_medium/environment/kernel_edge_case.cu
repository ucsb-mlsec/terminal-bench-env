#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <stdio.h>

namespace cg = cooperative_groups;

// Cooperative kernel that operates at maximum capacity limits
// This configuration tests the absolute boundary conditions for cooperative launches
__global__ void edgeCaseCooperativeKernel(int* data, int size) {
    // Get cooperative grid group handle
    cg::grid_group grid = cg::this_grid();
    
    // Calculate global thread index
    int globalThreadIdx = blockIdx.x * blockDim.x + threadIdx.x;
    int totalThreads = gridDim.x * blockDim.x;
    
    // Process data elements across all threads
    for (int i = globalThreadIdx; i < size; i += totalThreads) {
        data[i] = data[i] * 2;
    }
    
    // Synchronize all threads across the entire grid
    grid.sync();
    
    // Second phase computation after grid-wide synchronization
    for (int i = globalThreadIdx; i < size; i += totalThreads) {
        data[i] = data[i] + 1;
    }
}

int main() {
    // Edge case configuration: launching at exactly the maximum allowed limits
    // Grid dimensions: 32768 blocks - this equals the max cooperative blocks limit
    // Block dimensions: 1024 threads - this equals the max threads per block limit
    // Total thread count: 32768 * 1024 = 33,554,432 threads
    
    dim3 gridDim(32768, 1, 1);   // Exactly at max cooperative blocks boundary
    dim3 blockDim(1024, 1, 1);   // Exactly at max threads per block boundary
    
    int dataSize = 1000000;
    int* d_data;
    
    // Allocate device memory
    cudaMalloc(&d_data, dataSize * sizeof(int));
    
    // Initialize data
    int* h_data = new int[dataSize];
    for (int i = 0; i < dataSize; i++) {
        h_data[i] = i;
    }
    cudaMemcpy(d_data, h_data, dataSize * sizeof(int), cudaMemcpyHostToDevice);
    
    // Query device properties to verify cooperative launch support
    int device = 0;
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, device);
    
    // Verify cooperative launch capability
    if (!props.cooperativeLaunch) {
        printf("Device does not support cooperative launches\n");
        return -1;
    }
    
    // Setup launch parameters for cooperative kernel
    void* kernelArgs[] = { &d_data, &dataSize };
    
    // Launch cooperative kernel at maximum valid configuration
    // This is a boundary test: 32768 blocks = max, 1024 threads/block = max
    // Both values are AT the limit, not exceeding it, so this should be VALID
    cudaError_t result = cudaLaunchCooperativeKernel(
        (void*)edgeCaseCooperativeKernel,
        gridDim,
        blockDim,
        kernelArgs,
        0,  // Shared memory size
        0   // Stream
    );
    
    if (result != cudaSuccess) {
        printf("Cooperative kernel launch failed: %s\n", cudaGetErrorString(result));
        return -1;
    }
    
    // Wait for kernel completion
    cudaDeviceSynchronize();
    
    // Copy results back
    cudaMemcpy(h_data, d_data, dataSize * sizeof(int), cudaMemcpyDeviceToHost);
    
    // Verify results
    bool success = true;
    for (int i = 0; i < 100 && i < dataSize; i++) {
        int expected = i * 2 + 1;
        if (h_data[i] != expected) {
            success = false;
            break;
        }
    }
    
    printf("Edge case cooperative kernel (32768 blocks, 1024 threads/block): %s\n", 
           success ? "PASSED" : "FAILED");
    
    // Cleanup
    delete[] h_data;
    cudaFree(d_data);
    
    return 0;
}
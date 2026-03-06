#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <stdio.h>

// Cooperative kernel that requires grid-wide synchronization
__global__ void cooperativeKernel(int* data, int size) {
    namespace cg = cooperative_groups;
    
    // Get the grid group for cooperative synchronization
    cg::grid_group grid = cg::this_grid();
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < size) {
        data[idx] = idx * 2;
    }
    
    // Grid-wide synchronization - requires cooperative launch
    grid.sync();
    
    if (idx < size) {
        data[idx] += 1;
    }
}

int main() {
    int* d_data;
    int dataSize = 40000 * 128;
    
    cudaMalloc(&d_data, dataSize * sizeof(int));
    
    // Configure cooperative kernel launch
    // Using large grid dimensions for processing
    dim3 gridDim(40000, 1, 1);   // 40000 blocks - attempting large-scale parallelism
    dim3 blockDim(128, 1, 1);     // 128 threads per block
    
    void* kernelArgs[] = { &d_data, &dataSize };
    
    // Launch cooperative kernel with grid-wide synchronization capability
    cudaLaunchCooperativeKernel((void*)cooperativeKernel, gridDim, blockDim, kernelArgs, 0, 0);
    
    cudaDeviceSynchronize();
    
    cudaFree(d_data);
    
    return 0;
}
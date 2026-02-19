#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <stdio.h>

namespace cg = cooperative_groups;

// Cooperative kernel that uses grid-wide synchronization
__global__ void cooperativeKernel(int *data, int size) {
    cg::grid_group grid = cg::this_grid();
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < size) {
        data[idx] = idx * 2;
    }
    
    // Grid-wide synchronization - only possible with cooperative launch
    grid.sync();
    
    if (idx < size) {
        data[idx] += 1;
    }
}

int main() {
    int size = 65536;
    int *d_data;
    
    cudaMalloc(&d_data, size * sizeof(int));
    
    // Cooperative kernel launch configuration
    // Grid dimensions: 256 blocks (well below the typical max of 32768)
    dim3 gridDim(256, 1, 1);
    // Block dimensions: 256 threads per block
    dim3 blockDim(256, 1, 1);
    
    // Check cooperative launch support
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    
    if (prop.cooperativeLaunch) {
        void *kernelArgs[] = { &d_data, &size };
        
        // Launch cooperative kernel - requires special launch API
        cudaLaunchCooperativeKernel(
            (void*)cooperativeKernel,
            gridDim,
            blockDim,
            kernelArgs
        );
        
        cudaDeviceSynchronize();
        printf("Cooperative kernel launched successfully with 256 blocks\n");
    }
    
    cudaFree(d_data);
    return 0;
}
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <stdio.h>

using namespace cooperative_groups;

__global__ void prefixSumKernel(int* input, int* output, int n) {
    auto grid = this_grid();
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    // Copy input to output initially
    for (int i = tid; i < n; i += stride) {
        output[i] = input[i];
    }
    
    grid.sync();
    
    // Parallel prefix sum using grid-wide synchronization
    for (int offset = 1; offset < n; offset *= 2) {
        int temp = 0;
        if (tid >= offset && tid < n) {
            temp = output[tid - offset];
        }
        
        grid.sync();
        
        if (tid >= offset && tid < n) {
            output[tid] += temp;
        }
        
        grid.sync();
    }
}

void launchPrefixSum(int* d_input, int* d_output, int n) {
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;
    
    // Query device properties to ensure cooperative launch support
    int device;
    cudaGetDevice(&device);
    
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, device);
    
    if (!props.cooperativeLaunch) {
        printf("Device does not support cooperative kernel launch\n");
        return;
    }
    
    // Setup kernel arguments
    void* args[] = { &d_input, &d_output, &n };
    
    // Calculate maximum blocks that can be launched cooperatively
    int numBlocksPerSm = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &numBlocksPerSm,
        prefixSumKernel,
        blockSize,
        0
    );
    
    int maxBlocks = numBlocksPerSm * props.multiProcessorCount;
    if (numBlocks > maxBlocks) {
        numBlocks = maxBlocks;
    }
    
    dim3 gridSize(numBlocks);
    dim3 threadBlockSize(blockSize);
    
    // Launch using cooperative kernel API
    cudaLaunchCooperativeKernel(
        (void*)prefixSumKernel,
        gridSize,
        threadBlockSize,
        args
    );
    
    cudaDeviceSynchronize();
}
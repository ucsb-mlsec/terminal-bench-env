#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <stdio.h>

using namespace cooperative_groups;

__global__ void reductionCoopKernel(float* input, float* output, int n) {
    // Get grid group for grid-wide synchronization
    auto grid = this_grid();
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int gridSize = blockDim.x * gridDim.x;
    
    // Shared memory for block-level reduction
    extern __shared__ float sdata[];
    
    // Grid-stride loop for loading data
    float sum = 0.0f;
    for (int i = tid; i < n; i += gridSize) {
        sum += input[i];
    }
    
    // Block-level reduction
    sdata[threadIdx.x] = sum;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sdata[threadIdx.x] += sdata[threadIdx.x + s];
        }
        __syncthreads();
    }
    
    // Write block result to global memory
    if (threadIdx.x == 0) {
        output[blockIdx.x] = sdata[0];
    }
    
    // Grid-wide synchronization before final reduction
    grid.sync();
    
    // Final reduction across blocks using first block
    if (blockIdx.x == 0) {
        float blockSum = 0.0f;
        if (threadIdx.x < gridDim.x) {
            blockSum = output[threadIdx.x];
        }
        
        sdata[threadIdx.x] = blockSum;
        __syncthreads();
        
        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (threadIdx.x < s) {
                sdata[threadIdx.x] += sdata[threadIdx.x + s];
            }
            __syncthreads();
        }
        
        if (threadIdx.x == 0) {
            output[0] = sdata[0];
        }
    }
}

void launchReduction(float* d_input, float* d_output, int size) {
    int threadsPerBlock = 256;
    int numBlocks = 128;
    size_t sharedMem = threadsPerBlock * sizeof(float);
    
    // BUG: Using standard launch syntax for cooperative kernel
    reductionCoopKernel<<<numBlocks, threadsPerBlock, sharedMem>>>(d_input, d_output, size);
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Kernel launch failed: %s\n", cudaGetErrorString(err));
    }
}
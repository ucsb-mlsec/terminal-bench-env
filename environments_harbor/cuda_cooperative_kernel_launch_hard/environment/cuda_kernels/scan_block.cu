#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <stdio.h>

using namespace cooperative_groups;

// Block-level parallel scan kernel using cooperative groups
// This kernel performs an inclusive scan within each block
__global__ void scanBlockKernel(int* d_input, int* d_output, int size) {
    // Use block-level cooperative group (NOT grid-level)
    auto block = this_thread_block();
    
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Shared memory for block-level scan
    extern __shared__ int temp[];
    
    // Load input into shared memory
    if (gid < size) {
        temp[tid] = d_input[gid];
    } else {
        temp[tid] = 0;
    }
    
    block.sync();
    
    // Up-sweep phase (reduce)
    int offset = 1;
    for (int d = blockDim.x >> 1; d > 0; d >>= 1) {
        block.sync();
        if (tid < d) {
            int ai = offset * (2 * tid + 1) - 1;
            int bi = offset * (2 * tid + 2) - 1;
            temp[bi] += temp[ai];
        }
        offset *= 2;
    }
    
    // Clear the last element
    if (tid == 0) {
        temp[blockDim.x - 1] = 0;
    }
    
    block.sync();
    
    // Down-sweep phase
    for (int d = 1; d < blockDim.x; d *= 2) {
        offset >>= 1;
        block.sync();
        if (tid < d) {
            int ai = offset * (2 * tid + 1) - 1;
            int bi = offset * (2 * tid + 2) - 1;
            int t = temp[ai];
            temp[ai] = temp[bi];
            temp[bi] += t;
        }
    }
    
    block.sync();
    
    // Write results back to global memory
    if (gid < size) {
        d_output[gid] = temp[tid];
    }
}

// Host function to launch the kernel
void launchScanBlock(int* d_input, int* d_output, int size) {
    int threadsPerBlock = 256;
    int blocks = (size + threadsPerBlock - 1) / threadsPerBlock;
    int sharedMemSize = threadsPerBlock * sizeof(int);
    
    // Standard launch syntax - CORRECT for block-level cooperative groups
    scanBlockKernel<<<blocks, threadsPerBlock, sharedMemSize>>>(d_input, d_output, size);
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Kernel launch error: %s\n", cudaGetErrorString(err));
    }
}
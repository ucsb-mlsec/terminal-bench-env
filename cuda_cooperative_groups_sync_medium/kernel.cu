#include <cuda_runtime.h>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

__global__ void compute_kernel(float* input, float* output, float* grid_result, int n) {
    // Get cooperative group handles
    cg::thread_block block = cg::this_thread_block();
    cg::grid_group grid = cg::this_grid();
    
    // Shared memory for block-level operations
    __shared__ float shared_data[256];
    __shared__ float block_sum;
    
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Stage 1: Load data and write to shared memory
    if (gid < n) {
        shared_data[tid] = input[gid] * 2.0f;  // shared_memory_write
    }
    
    // Stage 2: Perform block-level reduction
    float local_sum = 0.0f;
    if (gid < n) {
        local_sum = shared_data[tid];  // shared_memory_read
    }
    
    // Block reduction without proper sync after shared memory write
    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (tid < offset) {
            local_sum += shared_data[tid + offset];  // block_reduction
            shared_data[tid] = local_sum;  // shared_memory_write
        }
        block.sync();
    }
    
    // Stage 3: Store block result
    if (tid == 0) {
        block_sum = shared_data[0];  // shared_memory_read
        output[blockIdx.x] = block_sum;
    }
    
    block.sync();
    
    // Stage 4: Grid-wide operation
    if (tid == 0) {
        atomicAdd(grid_result, block_sum);  // grid_reduction
    }
    
    // Stage 5: Another shared memory operation
    if (gid < n) {
        shared_data[tid] = output[blockIdx.x] + 1.0f;  // shared_memory_write
    }
    
    // Missing sync before read
    if (gid < n) {
        float temp = shared_data[tid];  // shared_memory_read
        input[gid] = temp * 0.5f;
    }
    
    // Stage 6: Grid-level broadcast (requires grid sync)
    if (blockIdx.x == 0 && tid == 0) {
        grid_result[1] = block_sum;  // grid_reduction
    }
    
    // Attempting to use result without grid sync
    if (gid < n) {
        output[gid] = input[gid] + grid_result[1];
    }
}
#include <cuda_runtime.h>
#include <stdio.h>

#define BLOCK_SIZE 256
#define WARP_SIZE 32

// Entry point: reduce_tree_kernel

__device__ void warp_reduce(volatile float* sdata, int tid) {
    sdata[tid] += sdata[tid + 32];
    sdata[tid] += sdata[tid + 16];
    sdata[tid] += sdata[tid + 8];
    sdata[tid] += sdata[tid + 4];
    sdata[tid] += sdata[tid + 2];
    sdata[tid] += sdata[tid + 1];
}

__global__ void reduce_partition_kernel(float* data, int start, int end, int depth);
__global__ void final_reduce_kernel(float* data, int size);

__global__ void reduce_tree_kernel(float* data, int size, int depth) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    
    __shared__ float shared_data[BLOCK_SIZE];
    
    if (tid < size) {
        shared_data[threadIdx.x] = data[tid];
    } else {
        shared_data[threadIdx.x] = 0.0f;
    }
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > WARP_SIZE; s >>= 1) {
        if (threadIdx.x < s) {
            shared_data[threadIdx.x] += shared_data[threadIdx.x + s];
        }
        __syncthreads();
    }
    
    if (threadIdx.x < WARP_SIZE) {
        warp_reduce(shared_data, threadIdx.x);
    }
    
    if (size > BLOCK_SIZE && depth < 22) {
        reduce_partition_kernel<<<4, BLOCK_SIZE>>>(data, 0, size, depth + 1);
    }
    
    if (threadIdx.x == 0) {
        data[blockIdx.x] = shared_data[0];
    }
}

__global__ void reduce_partition_kernel(float* data, int start, int end, int depth) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int range = end - start;
    int global_tid = start + tid;
    
    __shared__ float partial_sum[BLOCK_SIZE];
    
    if (tid < range && global_tid < end) {
        partial_sum[threadIdx.x] = data[global_tid];
    } else {
        partial_sum[threadIdx.x] = 0.0f;
    }
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s && (threadIdx.x + s) < BLOCK_SIZE) {
            partial_sum[threadIdx.x] += partial_sum[threadIdx.x + s];
        }
        __syncthreads();
    }
    
    if (range > BLOCK_SIZE * 2 && depth < 20) {
        int mid = start + range / 2;
        if (blockIdx.x == 0 && threadIdx.x == 0) {
            reduce_partition_kernel<<<2, BLOCK_SIZE>>>(data, start, mid, depth + 1);
            cudaDeviceSynchronize();
            
            reduce_partition_kernel<<<2, BLOCK_SIZE>>>(data, mid, end, depth + 1);
            cudaDeviceSynchronize();
        }
    }
    
    if (threadIdx.x == 0) {
        atomicAdd(&data[start], partial_sum[0]);
    }
}

__global__ void final_reduce_kernel(float* data, int size) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    
    __shared__ float reduction_buffer[BLOCK_SIZE];
    
    if (tid < size) {
        reduction_buffer[threadIdx.x] = data[tid];
    } else {
        reduction_buffer[threadIdx.x] = 0.0f;
    }
    __syncthreads();
    
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            reduction_buffer[threadIdx.x] += reduction_buffer[threadIdx.x + stride];
        }
        __syncthreads();
    }
    
    if (size > 1024 && blockIdx.x == 0 && threadIdx.x == 0) {
        reduce_partition_kernel<<<8, BLOCK_SIZE>>>(data, 0, size, 1);
        cudaDeviceSynchronize();
    }
    
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        data[0] = reduction_buffer[0];
    }
}

__global__ void hierarchical_sum_kernel(float* input, float* output, int level, int max_level) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (level < max_level) {
        output[tid] = input[tid] + input[tid + 1];
        
        if (level < max_level - 1) {
            hierarchical_sum_kernel<<<gridDim.x, blockDim.x>>>(output, output, level + 1, max_level);
        }
    }
}
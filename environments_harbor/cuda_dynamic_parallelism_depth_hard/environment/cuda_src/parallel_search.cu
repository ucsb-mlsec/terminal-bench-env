#include <cuda_runtime.h>
#include <stdio.h>

#define MAX_BINARY_DEPTH 16
#define MAX_FILTER_DEPTH 10

// Entry point: search_kernel

__device__ void check_element(int* data, int target, int index) {
    if (data[index] == target) {
        printf("Found target %d at index %d\n", target, index);
    }
}

__global__ void binary_search_kernel(int* data, int target, int start, int end, int depth);

__global__ void search_kernel(int* data, int target, int start, int end) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (tid == 0) {
        // Initialize search parameters
        int range = end - start;
        if (range > 0) {
            // Launch binary search kernel for subdividing the search space
            dim3 grid(1);
            dim3 block(1);
            binary_search_kernel<<<grid, block>>>(data, target, start, end, 0);
        }
    }
}

__global__ void binary_search_kernel(int* data, int target, int start, int end, int depth) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (tid != 0) return;
    
    // Check if we've reached maximum depth
    if (depth >= MAX_BINARY_DEPTH) {
        // Linear search at leaf level
        for (int i = start; i < end; i++) {
            check_element(data, target, i);
        }
        return;
    }
    
    int mid = start + (end - start) / 2;
    
    // Check middle element
    if (data[mid] == target) {
        printf("Found at index %d (depth %d)\n", mid, depth);
        return;
    }
    
    // Recursively search both halves
    if (mid > start) {
        dim3 grid(1);
        dim3 block(1);
        binary_search_kernel<<<grid, block>>>(data, target, start, mid, depth + 1);
    }
    
    if (mid + 1 < end) {
        dim3 grid(1);
        dim3 block(1);
        binary_search_kernel<<<grid, block>>>(data, target, mid + 1, end, depth + 1);
    }
}

__device__ bool passes_filter(int value, int threshold) {
    return value > threshold;
}

__global__ void filter_subdivide_kernel(int* data, int start, int end, int currentDepth);

// Entry point: parallel_filter_kernel
__global__ void parallel_filter_kernel(int* data, int* output, int size, int filterDepth) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (tid == 0) {
        // Initialize filtering operation
        int chunkSize = size / 4;
        if (chunkSize > 0) {
            // Start hierarchical filtering
            dim3 grid(1);
            dim3 block(32);
            filter_subdivide_kernel<<<grid, block>>>(data, 0, size, 0);
        }
    }
    
    // Perform local filtering
    if (tid < size) {
        if (passes_filter(data[tid], 100)) {
            output[tid] = data[tid];
        } else {
            output[tid] = 0;
        }
    }
}

__global__ void filter_subdivide_kernel(int* data, int start, int end, int currentDepth) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (tid != 0) return;
    
    // Check depth limit
    if (currentDepth >= MAX_FILTER_DEPTH) {
        // Process at leaf level
        for (int i = start; i < end; i++) {
            if (passes_filter(data[i], 50)) {
                atomicAdd(&data[0], 1);
            }
        }
        return;
    }
    
    int range = end - start;
    if (range <= 1) {
        return;
    }
    
    int quarter = range / 4;
    
    // Subdivide into 4 parts for parallel processing
    if (quarter > 0) {
        dim3 grid(1);
        dim3 block(1);
        filter_subdivide_kernel<<<grid, block>>>(data, start, start + quarter, currentDepth + 1);
    }
}

__global__ void simple_map_kernel(int* data, int size) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (tid < size) {
        data[tid] = data[tid] * 2;
    }
}

void launch_search(int* d_data, int target, int size) {
    dim3 grid(1);
    dim3 block(256);
    search_kernel<<<grid, block>>>(d_data, target, 0, size);
}

void launch_filter(int* d_data, int* d_output, int size) {
    dim3 grid((size + 255) / 256);
    dim3 block(256);
    parallel_filter_kernel<<<grid, block>>>(d_data, d_output, size, 0);
}
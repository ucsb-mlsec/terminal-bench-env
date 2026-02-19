#include <cuda_runtime.h>
#include <stdio.h>

// Entry point - launched from host
__global__ void initializeData(int *data, int size, int level) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < size) {
        data[idx] = idx * 2;
    }
    
    if (level < 5) {
        // Launch preprocessing kernel
        if (threadIdx.x == 0 && blockIdx.x == 0) {
            int blocks = (size + 255) / 256;
            preprocessData<<<blocks, 256>>>(data, size, level + 1);
        }
    }
}

__global__ void preprocessData(int *data, int size, int level) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < size) {
        data[idx] += 10;
    }
    
    if (level < 8) {
        if (threadIdx.x == 0 && blockIdx.x == 0) {
            int blocks = (size + 255) / 256;
            computeStep1<<<blocks, 256>>>(data, size, level + 1);
        }
    }
}

__global__ void computeStep1(int *data, int size, int level) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < size) {
        data[idx] = data[idx] * 3;
    }
    
    if (level < 12) {
        if (threadIdx.x == 0 && blockIdx.x == 0) {
            int blocks = (size + 255) / 256;
            
            // Conditional launch based on level
            if (level % 2 == 0) {
                computeStep2<<<blocks, 256>>>(data, size, level + 1);
            } else {
                alternateCompute<<<blocks, 256>>>(data, size, level + 1);
            }
        }
    }
}

__global__ void computeStep2(int *data, int size, int level) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < size) {
        data[idx] = data[idx] + idx;
    }
    
    if (level < 18) {
        if (threadIdx.x == 0 && blockIdx.x == 0) {
            int blocks = (size + 255) / 256;
            finalizeCompute<<<blocks, 256>>>(data, size, level + 1);
        }
    }
}

__global__ void alternateCompute(int *data, int size, int level) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < size) {
        data[idx] = data[idx] - idx;
    }
    
    if (level < 15) {
        if (threadIdx.x == 0 && blockIdx.x == 0) {
            int blocks = (size + 255) / 256;
            finalizeCompute<<<blocks, 256>>>(data, size, level + 1);
        }
    }
}

__global__ void finalizeCompute(int *data, int size, int level) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < size) {
        data[idx] = data[idx] / 2;
    }
    
    // No further launches - end of chain
}

// Entry point - launched from host
__global__ void processMain(float *input, float *output, int count, int depth) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < count) {
        output[idx] = input[idx] * 2.0f;
    }
    
    if (depth < 20) {
        if (threadIdx.x == 0 && blockIdx.x == 0) {
            int blocks = (count + 255) / 256;
            
            // Multiple conditional launches
            if (depth < 10) {
                transformData<<<blocks, 256>>>(input, output, count, depth + 1);
            } else if (depth < 15) {
                reduceData<<<blocks, 256>>>(input, output, count, depth + 1);
            } else {
                aggregateResults<<<blocks, 256>>>(input, output, count, depth + 1);
            }
        }
    }
}

__global__ void transformData(float *input, float *output, int count, int depth) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < count) {
        output[idx] = input[idx] + 5.0f;
    }
    
    if (depth < 25) {
        if (threadIdx.x == 0 && blockIdx.x == 0) {
            int blocks = (count + 255) / 256;
            reduceData<<<blocks, 256>>>(input, output, count, depth + 1);
        }
    }
}

__global__ void reduceData(float *input, float *output, int count, int depth) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < count) {
        output[idx] = output[idx] * 0.5f;
    }
    
    if (depth < 30) {
        if (threadIdx.x == 0 && blockIdx.x == 0) {
            int blocks = (count + 255) / 256;
            aggregateResults<<<blocks, 256>>>(input, output, count, depth + 1);
        }
    }
}

__global__ void aggregateResults(float *input, float *output, int count, int depth) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < count) {
        output[idx] = (input[idx] + output[idx]) / 2.0f;
    }
    
    // Terminal kernel - no further launches
}
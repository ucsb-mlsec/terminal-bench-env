#include <cuda_runtime.h>
#include <stdio.h>

// Processing kernels - create complex call chains

__global__ void processLevel5(float* data, int size, int depth) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] = data[idx] * 1.05f + depth;
    }
}

__global__ void processLevel4(float* data, int size, int depth) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] = data[idx] * 2.0f;
    }
    
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        dim3 grid((size + 255) / 256);
        dim3 block(256);
        processLevel5<<<grid, block>>>(data, size, depth + 1);
    }
}

__global__ void processLevel3(float* data, int size, int depth) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] = data[idx] + 3.0f;
    }
    
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        dim3 grid((size + 255) / 256);
        dim3 block(256);
        processLevel4<<<grid, block>>>(data, size, depth + 1);
    }
}

__global__ void transformData(float* input, float* output, int size, int iteration) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = input[idx] * 1.5f + iteration * 0.1f;
    }
}

__global__ void reduceResults(float* data, int size, int depth) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size / 2) {
        data[idx] = data[idx] + data[idx + size / 2];
    }
    
    if (threadIdx.x == 0 && blockIdx.x == 0 && depth < 10) {
        dim3 grid(1);
        dim3 block(256);
        reduceResults<<<grid, block>>>(data, size / 2, depth + 1);
    }
}

__global__ void filterData(float* data, int size, float threshold, int depth) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        if (data[idx] < threshold) {
            data[idx] = 0.0f;
        }
    }
    
    if (threadIdx.x == 0 && blockIdx.x == 0 && depth < 8) {
        dim3 grid((size + 255) / 256);
        dim3 block(256);
        transformData<<<grid, block>>>(data, data, size, depth);
        
        if (depth < 5) {
            reduceResults<<<dim3(1), dim3(256)>>>(data, size, depth + 1);
        }
    }
}

__global__ void processLevel2(float* data, int size, int depth) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] = data[idx] * 1.2f;
    }
    
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        dim3 grid((size + 255) / 256);
        dim3 block(256);
        
        processLevel3<<<grid, block>>>(data, size, depth + 1);
        
        if (depth < 6) {
            filterData<<<grid, block>>>(data, size, 0.5f, depth + 1);
        }
    }
}

__global__ void processLevel1(float* data, int size, int depth) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] = data[idx] + 1.0f;
    }
    
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        dim3 grid((size + 255) / 256);
        dim3 block(256);
        
        processLevel2<<<grid, block>>>(data, size, depth + 1);
        
        if (depth < 4) {
            transformData<<<grid, block>>>(data, data, size, depth);
        }
    }
}

__global__ void recursiveProcess(float* data, int size, int depth, int maxDepth) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] = data[idx] * 0.99f;
    }
    
    if (threadIdx.x == 0 && blockIdx.x == 0 && depth < maxDepth) {
        dim3 grid((size + 255) / 256);
        dim3 block(256);
        recursiveProcess<<<grid, block>>>(data, size, depth + 1, maxDepth);
    }
}

__global__ void advancedFilter(float* data, int size, int depth) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] = (data[idx] > 1.0f) ? data[idx] : 1.0f;
    }
    
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        dim3 grid((size + 255) / 256);
        dim3 block(256);
        
        processLevel1<<<grid, block>>>(data, size, depth + 1);
        
        if (depth < 15) {
            recursiveProcess<<<grid, block>>>(data, size, depth + 1, depth + 12);
        }
    }
}
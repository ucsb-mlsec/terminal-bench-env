#include <cuda_runtime.h>
#include <stdio.h>

#define RECURSION_LIMIT 8
#define LOOP_ITERATIONS 3

// Forward declarations
__global__ void computePhase1(float* data, int size, int depth);
__global__ void computePhase2(float* data, int size, int depth);
__global__ void computePhase3(float* data, int size, int depth);
__global__ void calculateResults(float* data, int size, int depth);
__global__ void aggregateData(float* data, int size, int depth);
__global__ void finalizeCompute(float* data, int size, int depth);
__global__ void recursiveProcess(float* data, int size, int depth, int maxDepth);

// Entry point kernel - launches the computation pipeline
// Launched from host code at depth 0
__global__ void computePhase1(float* data, int size, int depth) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < size) {
        data[idx] = data[idx] * 1.5f + 0.5f;
    }
    
    // Launch phase 2 from first thread
    if (idx == 0) {
        dim3 grid((size + 255) / 256);
        dim3 block(256);
        computePhase2<<<grid, block>>>(data, size, depth + 1);
    }
}

// Phase 2: Intermediate processing
__global__ void computePhase2(float* data, int size, int depth) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < size) {
        data[idx] = sqrtf(data[idx] * data[idx] + 1.0f);
    }
    
    // Launch phase 3 from first thread
    if (idx == 0) {
        dim3 grid((size + 255) / 256);
        dim3 block(256);
        computePhase3<<<grid, block>>>(data, size, depth + 1);
    }
}

// Phase 3: More complex processing with loop-based launches
__global__ void computePhase3(float* data, int size, int depth) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < size) {
        data[idx] = data[idx] * 2.0f - 1.0f;
    }
    
    // Launch calculateResults multiple times in a loop
    if (idx == 0) {
        dim3 grid((size + 255) / 256);
        dim3 block(256);
        
        for (int i = 0; i < LOOP_ITERATIONS; i++) {
            calculateResults<<<grid, block>>>(data, size, depth + 1);
            cudaDeviceSynchronize();
        }
    }
}

// Calculate results: performs calculations and launches aggregation
__global__ void calculateResults(float* data, int size, int depth) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < size) {
        data[idx] = data[idx] / (1.0f + data[idx]);
    }
    
    // Launch aggregation phase
    if (idx == 0) {
        dim3 grid((size + 255) / 256);
        dim3 block(256);
        aggregateData<<<grid, block>>>(data, size, depth + 1);
    }
}

// Aggregate data: combines results and launches recursive processing
__global__ void aggregateData(float* data, int size, int depth) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < size) {
        float sum = 0.0f;
        if (idx > 0) sum += data[idx - 1];
        sum += data[idx];
        if (idx < size - 1) sum += data[idx + 1];
        data[idx] = sum / 3.0f;
    }
    
    // Launch recursive processing
    if (idx == 0) {
        dim3 grid((size + 255) / 256);
        dim3 block(256);
        recursiveProcess<<<grid, block>>>(data, size, depth + 1, RECURSION_LIMIT);
    }
}

// Recursive kernel: calls itself until reaching depth limit
__global__ void recursiveProcess(float* data, int size, int depth, int maxDepth) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < size) {
        data[idx] = data[idx] * 0.95f + 0.05f;
    }
    
    // Recursive call with depth check
    if (idx == 0 && depth < maxDepth) {
        dim3 grid((size + 255) / 256);
        dim3 block(256);
        recursiveProcess<<<grid, block>>>(data, size, depth + 1, maxDepth);
    } else if (idx == 0 && depth >= maxDepth) {
        // Base case: launch finalization
        dim3 grid((size + 255) / 256);
        dim3 block(256);
        finalizeCompute<<<grid, block>>>(data, size, depth + 1);
    }
}

// Finalize computation: final processing step
__global__ void finalizeCompute(float* data, int size, int depth) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < size) {
        data[idx] = data[idx] * 1.1f;
        
        // Apply threshold
        if (data[idx] > 10.0f) {
            data[idx] = 10.0f;
        } else if (data[idx] < -10.0f) {
            data[idx] = -10.0f;
        }
    }
}

// Host function to launch the pipeline
extern "C" void launchComputePipeline(float* d_data, int size) {
    dim3 grid((size + 255) / 256);
    dim3 block(256);
    
    // Entry point: launch from host at depth 0
    computePhase1<<<grid, block>>>(d_data, size, 0);
    cudaDeviceSynchronize();
}
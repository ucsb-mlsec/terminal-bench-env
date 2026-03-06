#include <cuda_runtime.h>
#include <stdio.h>

#define MAX_RECURSIVE_DEPTH 12
#define CASCADE_DEPTH 8
#define CHAIN_LIMIT 10

// Forward declarations for cross-file dependencies
extern __global__ void matrixKernel(int depth);
extern __global__ void dataProcessor(int level, int* data);
extern __global__ void utilityKernel(int stage);

// Deep recursive kernel - can go 12 levels deep
__global__ void deepRecursive(int depth, int* counter) {
    if (threadIdx.x == 0) {
        if (depth < MAX_RECURSIVE_DEPTH) {
            // Recursive call with increased depth
            deepRecursive<<<1, 32>>>(depth + 1, counter);
            cudaDeviceSynchronize();
            
            // Also launch nested operation for added complexity
            if (depth < 8) {
                extern __global__ void nestedOperation(int level, int branch);
                nestedOperation<<<1, 16>>>(depth + 1, 1);
            }
        }
        atomicAdd(counter, 1);
    }
}

// Nested operation with multiple branching paths
__global__ void nestedOperation(int level, int branch) {
    if (threadIdx.x == 0 && level < 10) {
        if (branch == 1) {
            // Path 1: Launch advanced process
            advancedProcess<<<1, 32>>>(level + 1, branch * 2);
            cudaDeviceSynchronize();
        } else if (branch == 2) {
            // Path 2: Launch multi-level compute
            multiLevelCompute<<<1, 16>>>(level + 1);
            cudaDeviceSynchronize();
        } else {
            // Path 3: Launch cascade kernel
            cascadeKernel<<<1, 32>>>(level + 1, 0);
            cudaDeviceSynchronize();
        }
        
        // Additional cross-file call
        if (level < 6) {
            matrixKernel<<<1, 64>>>(level + 1);
            cudaDeviceSynchronize();
        }
    }
}

// Advanced process with deep chaining
__global__ void advancedProcess(int level, int mode) {
    if (threadIdx.x == 0) {
        if (level < 15) {
            // Launch multiple kernels based on mode
            if (mode % 4 == 0) {
                chainedExecution<<<1, 32>>>(level + 1, mode / 2);
                cudaDeviceSynchronize();
            }
            
            if (mode % 3 == 0) {
                multiLevelCompute<<<1, 16>>>(level + 1);
                cudaDeviceSynchronize();
            }
            
            // Deep chain continuation
            if (level < 12) {
                cascadeKernel<<<1, 32>>>(level + 1, mode);
                cudaDeviceSynchronize();
            }
            
            // Cross-file dependency
            if (level < 8) {
                dataProcessor<<<1, 64>>>(level + 1, nullptr);
                cudaDeviceSynchronize();
            }
        }
    }
}

// Multi-level compute with recursive pattern
__global__ void multiLevelCompute(int level) {
    if (threadIdx.x == 0 && level < CHAIN_LIMIT) {
        // Self-recursive call
        if (level < 9) {
            multiLevelCompute<<<1, 32>>>(level + 1);
            cudaDeviceSynchronize();
        }
        
        // Branch to other kernels
        if (level % 2 == 0 && level < 7) {
            cascadeKernel<<<1, 16>>>(level + 1, level);
            cudaDeviceSynchronize();
            
            // Additional deep path
            chainedExecution<<<1, 32>>>(level + 1, level * 2);
            cudaDeviceSynchronize();
        }
        
        // Cross-file call for deeper nesting
        if (level < 5) {
            utilityKernel<<<1, 32>>>(level + 1);
            cudaDeviceSynchronize();
        }
    }
}

// Cascade kernel with multiple launch points
__global__ void cascadeKernel(int level, int param) {
    if (threadIdx.x == 0) {
        if (level < CASCADE_DEPTH) {
            // Primary cascade path
            cascadeKernel<<<1, 32>>>(level + 1, param + 1);
            cudaDeviceSynchronize();
        }
        
        // Secondary paths creating deep nesting
        if (level < 6 && param % 2 == 0) {
            chainedExecution<<<1, 16>>>(level + 1, param);
            cudaDeviceSynchronize();
            
            nestedOperation<<<1, 32>>>(level + 1, 3);
            cudaDeviceSynchronize();
        }
        
        // Tertiary path for maximum depth
        if (level < 5) {
            advancedProcess<<<1, 32>>>(level + 1, param * 3);
            cudaDeviceSynchronize();
        }
    }
}

// Chained execution with longest possible path
__global__ void chainedExecution(int level, int chain_id) {
    if (threadIdx.x == 0 && level < 20) {
        // Create very deep chains
        if (chain_id < 15) {
            chainedExecution<<<1, 32>>>(level + 1, chain_id + 1);
            cudaDeviceSynchronize();
        }
        
        // Branch to create alternate deep paths
        if (level < 10 && chain_id % 3 == 0) {
            multiLevelCompute<<<1, 16>>>(level + 1);
            cudaDeviceSynchronize();
        }
        
        // Another branching path
        if (level < 8 && chain_id % 5 == 0) {
            deepRecursive<<<1, 32>>>(level + 1, nullptr);
            cudaDeviceSynchronize();
            
            cascadeKernel<<<1, 32>>>(level + 1, chain_id);
            cudaDeviceSynchronize();
        }
        
        // Cross-file for added depth
        if (level < 6) {
            matrixKernel<<<1, 64>>>(level + 1);
            cudaDeviceSynchronize();
        }
    }
}
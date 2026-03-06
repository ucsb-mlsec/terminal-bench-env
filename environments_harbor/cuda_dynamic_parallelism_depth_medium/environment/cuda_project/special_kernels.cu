#include <cuda_runtime.h>
#include <stdio.h>

// Forward declarations from other files
__global__ void recursiveProcess(int depth, int maxDepth);
__global__ void chainedCompute(int level);
__global__ void advancedOperation(int iteration);

// Entry point - launched from host code
__global__ void deepEntryPoint(int startLevel) {
    if (threadIdx.x == 0) {
        printf("Starting deep entry point at level %d\n", startLevel);
        
        // Launch specialProcess which will create a long chain
        specialProcess<<<1, 32>>>(startLevel + 1, 25);
        cudaDeviceSynchronize();
    }
}

__global__ void specialProcess(int currentLevel, int targetDepth) {
    if (threadIdx.x == 0 && currentLevel < targetDepth) {
        printf("Special process at level %d\n", currentLevel);
        
        // Launch extendedChain to continue building depth
        extendedChain<<<1, 16>>>(currentLevel + 1);
        cudaDeviceSynchronize();
        
        // Also launch maximizeDepth for parallel deep path
        maximizeDepth<<<1, 8>>>(currentLevel + 1, targetDepth);
        cudaDeviceSynchronize();
    }
}

__global__ void extendedChain(int depth) {
    if (threadIdx.x == 0) {
        printf("Extended chain at depth %d\n", depth);
        
        // Launch recursiveProcess from compute_kernels.cu
        recursiveProcess<<<1, 32>>>(depth + 1, 28);
        cudaDeviceSynchronize();
        
        // Additional launch to chainedCompute
        if (depth < 23) {
            chainedCompute<<<1, 16>>>(depth + 1);
            cudaDeviceSynchronize();
        }
    }
}

__global__ void maximizeDepth(int level, int maxLevel) {
    if (threadIdx.x == 0 && level < maxLevel) {
        printf("Maximize depth at level %d\n", level);
        
        // Continue the chain with advancedOperation
        advancedOperation<<<1, 32>>>(level + 1);
        cudaDeviceSynchronize();
        
        // Launch deepRecursive to push boundaries
        deepRecursive<<<1, 8>>>(level + 1, maxLevel);
        cudaDeviceSynchronize();
        
        // Launch continueCascade for additional depth
        continueCascade<<<1, 16>>>(level + 1);
        cudaDeviceSynchronize();
    }
}

__global__ void deepRecursive(int currentDepth, int limit) {
    if (threadIdx.x == 0 && currentDepth < limit) {
        printf("Deep recursive at depth %d\n", currentDepth);
        
        // Self-recursive call to build extreme depth
        if (currentDepth < 26) {
            deepRecursive<<<1, 4>>>(currentDepth + 1, limit);
            cudaDeviceSynchronize();
        }
        
        // Launch finalDepthKernel to maximize nesting
        finalDepthKernel<<<1, 8>>>(currentDepth + 1);
        cudaDeviceSynchronize();
    }
}

__global__ void continueCascade(int depth) {
    if (threadIdx.x == 0) {
        printf("Continue cascade at depth %d\n", depth);
        
        // Launch multiple kernels in sequence
        cascadeLevel1<<<1, 16>>>(depth + 1);
        cudaDeviceSynchronize();
        
        cascadeLevel2<<<1, 8>>>(depth + 1);
        cudaDeviceSynchronize();
    }
}

__global__ void cascadeLevel1(int level) {
    if (threadIdx.x == 0 && level < 24) {
        printf("Cascade level 1 at %d\n", level);
        
        // Launch cascadeLevel2 to continue
        cascadeLevel2<<<1, 8>>>(level + 1);
        cudaDeviceSynchronize();
        
        // Launch finalDepthKernel
        finalDepthKernel<<<1, 4>>>(level + 1);
        cudaDeviceSynchronize();
    }
}

__global__ void cascadeLevel2(int level) {
    if (threadIdx.x == 0 && level < 25) {
        printf("Cascade level 2 at %d\n", level);
        
        // Launch finalDepthKernel to reach maximum depth
        finalDepthKernel<<<1, 4>>>(level + 1);
        cudaDeviceSynchronize();
        
        // Launch terminalKernel for final depth push
        terminalKernel<<<1, 2>>>(level + 1);
        cudaDeviceSynchronize();
    }
}

__global__ void finalDepthKernel(int depth) {
    if (threadIdx.x == 0 && depth < 26) {
        printf("Final depth kernel at %d\n", depth);
        
        // Launch terminalKernel to complete deep chain
        terminalKernel<<<1, 2>>>(depth + 1);
        cudaDeviceSynchronize();
    }
}

__global__ void terminalKernel(int finalDepth) {
    if (threadIdx.x == 0) {
        printf("Terminal kernel reached at depth %d\n", finalDepth);
        // This is designed to be near or at the depth limit
    }
}
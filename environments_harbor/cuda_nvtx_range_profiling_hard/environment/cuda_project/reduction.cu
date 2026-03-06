#include <cuda_runtime.h>
#include <nvToolsExt.h>
#include <stdio.h>

#define BLOCK_SIZE 256

// CUDA kernel for parallel reduction (sum)
__global__ void reductionKernel(float *input, float *output, int n) {
    __shared__ float sdata[BLOCK_SIZE];
    
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Load input into shared memory
    sdata[tid] = (i < n) ? input[i] : 0.0f;
    __syncthreads();
    
    // Perform reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    // Write result for this block to global memory
    if (tid == 0) {
        output[blockIdx.x] = sdata[0];
    }
}

// Host function to perform reduction
float performReduction(float *d_input, int n) {
    float *d_output;
    float *h_output;
    int numBlocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    cudaMalloc(&d_output, numBlocks * sizeof(float));
    h_output = (float*)malloc(numBlocks * sizeof(float));
    
    // Launch reduction kernel with NVTX instrumentation
    nvtxRangePush("Reduction Kernel");
    reductionKernel<<<numBlocks, BLOCK_SIZE>>>(d_input, d_output, n);
    cudaDeviceSynchronize();
    
    // Copy results back to host
    nvtxRangePush("Copy Results");
    cudaMemcpy(h_output, d_output, numBlocks * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Final reduction on CPU
    float result = 0.0f;
    for (int i = 0; i < numBlocks; i++) {
        result += h_output[i];
    }
    
    // Missing nvtxRangePop - only one pop for two pushes (UNBALANCED)
    nvtxRangePop();
    
    cudaFree(d_output);
    free(h_output);
    
    return result;
}

// Main function for testing
int main() {
    const int N = 1024 * 1024;
    float *h_input, *d_input;
    
    // Allocate host memory
    h_input = (float*)malloc(N * sizeof(float));
    
    // Initialize input data
    for (int i = 0; i < N; i++) {
        h_input[i] = 1.0f;
    }
    
    // Allocate device memory
    cudaMalloc(&d_input, N * sizeof(float));
    cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice);
    
    // Perform reduction
    float sum = performReduction(d_input, N);
    
    printf("Sum: %f\n", sum);
    printf("Expected: %d\n", N);
    
    // Cleanup
    cudaFree(d_input);
    free(h_input);
    
    return 0;
}
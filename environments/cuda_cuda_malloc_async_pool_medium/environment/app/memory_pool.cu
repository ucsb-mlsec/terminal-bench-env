#include <cuda_runtime.h>
#include <stdio.h>
#include <stdint.h>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// Kernel to process data
__global__ void processData(float* data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] = data[idx] * 2.0f + 1.0f;
    }
}

int main() {
    // Initialize CUDA device
    int device = 0;
    CUDA_CHECK(cudaSetDevice(device));
    
    // Create CUDA stream for async operations
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    
    // Get the default memory pool for the device
    cudaMemPool_t memPool;
    CUDA_CHECK(cudaDeviceGetDefaultMemPool(&memPool, device));
    
    printf("Configuring memory pool...\n");
    
    // CRITICAL BUG: Setting release threshold to 0 causes aggressive memory release
    // This forces the pool to release memory back to the OS immediately after free
    // causing frequent reallocations and performance degradation
    uint64_t threshold = 0;
    CUDA_CHECK(cudaMemPoolSetAttribute(memPool, cudaMemPoolAttrReleaseThreshold, &threshold));
    
    // Set reserved memory for the pool
    uint64_t reserved = 128 * 1024 * 1024; // 128 MB
    CUDA_CHECK(cudaMemPoolSetAttribute(memPool, cudaMemPoolAttrReservedMemCurrent, &reserved));
    
    printf("Memory pool configured. Starting allocations...\n");
    
    const int numIterations = 100;
    const int dataSize = 1024 * 1024; // 1M floats
    const size_t bytes = dataSize * sizeof(float);
    
    // Simulate workload with repeated async allocations
    for (int i = 0; i < numIterations; i++) {
        float* d_data;
        
        // Async allocation from memory pool
        CUDA_CHECK(cudaMallocAsync(&d_data, bytes, stream));
        
        // Launch kernel to process data
        int blockSize = 256;
        int numBlocks = (dataSize + blockSize - 1) / blockSize;
        processData<<<numBlocks, blockSize, 0, stream>>>(d_data, dataSize);
        
        // Async free back to memory pool
        CUDA_CHECK(cudaFreeAsync(d_data, stream));
        
        if (i % 10 == 0) {
            printf("Completed iteration %d/%d\n", i, numIterations);
        }
    }
    
    // Synchronize to ensure all operations complete
    CUDA_CHECK(cudaStreamSynchronize(stream));
    
    printf("All operations completed.\n");
    
    // Cleanup
    CUDA_CHECK(cudaStreamDestroy(stream));
    
    return 0;
}
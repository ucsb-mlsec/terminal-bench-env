#include <cuda_runtime.h>
#include <stdio.h>

#define HISTOGRAM_SIZE 256
#define BLOCK_SIZE 256

// CUDA kernel for histogram computation using shared memory
__global__ void histogramKernel(const unsigned char *data, int *histogram, int dataSize) {
    __shared__ int localHist[HISTOGRAM_SIZE];
    
    int tid = threadIdx.x;
    int globalIdx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    // Initialize shared memory histogram
    if (tid < HISTOGRAM_SIZE) {
        localHist[tid] = 0;
    }
    __syncthreads();
    
    // Compute local histogram using atomic operations
    for (int i = globalIdx; i < dataSize; i += stride) {
        unsigned char value = data[i];
        atomicAdd(&localHist[value], 1);
    }
    __syncthreads();
    
    // Write local histogram to global memory
    if (tid < HISTOGRAM_SIZE) {
        atomicAdd(&histogram[tid], localHist[tid]);
    }
}

// Host function to compute histogram on GPU
void computeHistogramGPU(const unsigned char *hostData, int dataSize, int *hostHistogram) {
    unsigned char *deviceData;
    int *deviceHistogram;
    
    size_t dataBytes = dataSize * sizeof(unsigned char);
    size_t histBytes = HISTOGRAM_SIZE * sizeof(int);
    
    // Allocate device memory
    cudaMalloc((void**)&deviceData, dataBytes);
    cudaMalloc((void**)&deviceHistogram, histBytes);
    
    // Initialize histogram to zero
    cudaMemset(deviceHistogram, 0, histBytes);
    
    // Copy input data to device
    cudaMemcpy(deviceData, hostData, dataBytes, cudaMemcpyHostToDevice);
    
    // Launch kernel
    int numBlocks = (dataSize + BLOCK_SIZE - 1) / BLOCK_SIZE;
    if (numBlocks > 512) numBlocks = 512;
    
    histogramKernel<<<numBlocks, BLOCK_SIZE>>>(deviceData, deviceHistogram, dataSize);
    
    // Copy result back to host
    cudaMemcpy(hostHistogram, deviceHistogram, histBytes, cudaMemcpyDeviceToHost);
    
    // Synchronize
    cudaDeviceSynchronize();
    
    // Free device memory
    cudaFree(deviceData);
    cudaFree(deviceHistogram);
}

// Alternative kernel without shared memory for small datasets
__global__ void histogramKernelSimple(const unsigned char *data, int *histogram, int dataSize) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < dataSize) {
        unsigned char value = data[idx];
        atomicAdd(&histogram[value], 1);
    }
}

// Helper function to validate histogram results
bool validateHistogram(const int *histogram, int expectedTotal) {
    int sum = 0;
    for (int i = 0; i < HISTOGRAM_SIZE; i++) {
        sum += histogram[i];
    }
    return (sum == expectedTotal);
}
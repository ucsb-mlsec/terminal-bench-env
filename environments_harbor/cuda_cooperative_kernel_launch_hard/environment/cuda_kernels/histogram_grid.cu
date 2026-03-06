#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <stdio.h>

using namespace cooperative_groups;

// Histogram kernel using grid-wide cooperative synchronization
// This kernel computes a histogram across the entire grid using
// cooperative groups for grid-level synchronization
__global__ void histogramGridKernel(int* data, int* histogram, int size, int numBins) {
    // Get grid-level cooperative group for grid-wide synchronization
    grid_group g = this_grid();
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int gridSize = gridDim.x * blockDim.x;
    
    // Shared memory for local histogram bins per block
    extern __shared__ int localHist[];
    
    // Initialize local histogram
    for (int i = threadIdx.x; i < numBins; i += blockDim.x) {
        localHist[i] = 0;
    }
    __syncthreads();
    
    // First pass: compute local histograms
    for (int i = tid; i < size; i += gridSize) {
        int bin = data[i] % numBins;
        atomicAdd(&localHist[bin], 1);
    }
    __syncthreads();
    
    // Grid-wide synchronization to ensure all blocks finish local computation
    g.sync();
    
    // Second pass: aggregate local histograms to global histogram
    // Only first block aggregates results
    if (blockIdx.x == 0) {
        for (int i = threadIdx.x; i < numBins; i += blockDim.x) {
            atomicAdd(&histogram[i], localHist[i]);
        }
    }
    
    // Another grid-wide sync before reading final results
    g.sync();
    
    // All threads can now safely read the complete histogram
    if (tid == 0) {
        printf("Grid-wide histogram computation complete\n");
    }
}

// Host function to launch the histogram kernel
void computeHistogram(int* d_data, int* d_histogram, int dataSize, int numBins) {
    int threadsPerBlock = 256;
    int numBlocks = (dataSize + threadsPerBlock - 1) / threadsPerBlock;
    
    // Initialize histogram to zero
    cudaMemset(d_histogram, 0, numBins * sizeof(int));
    
    // Launch kernel with standard syntax
    // This performs grid-wide operations across all blocks
    size_t sharedMemSize = numBins * sizeof(int);
    histogramGridKernel<<<numBlocks, threadsPerBlock, sharedMemSize>>>(
        d_data, d_histogram, dataSize, numBins);
    
    cudaDeviceSynchronize();
}

// Wrapper function for external calls
void launchHistogramComputation(int* d_data, int* d_histogram, int dataSize) {
    const int NUM_BINS = 256;
    computeHistogram(d_data, d_histogram, dataSize, NUM_BINS);
}
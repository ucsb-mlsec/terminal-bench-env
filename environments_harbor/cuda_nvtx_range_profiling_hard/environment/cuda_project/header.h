#ifndef HEADER_H
#define HEADER_H

#include <cuda_runtime.h>
#include <stddef.h>

// Constants
#define MAX_BLOCK_SIZE 256
#define WARP_SIZE 32
#define MAX_GRID_DIM 65535
#define DEFAULT_STREAM 0

// Data structure definitions
typedef struct {
    float* data;
    size_t width;
    size_t height;
    size_t pitch;
} Matrix;

typedef struct {
    int* indices;
    float* values;
    size_t nnz;
    size_t rows;
    size_t cols;
} SparseMatrix;

// CUDA kernel declarations
__global__ void vectorAdd(const float* a, const float* b, float* c, int n);
__global__ void matrixMultiply(const float* A, const float* B, float* C, int m, int n, int k);
__global__ void reduceSum(const float* input, float* output, int n);

// Host function declarations
void initializeData(float* data, size_t size);
void processDataset(const float* input, float* output, size_t n);
int validateResults(const float* computed, const float* expected, size_t n, float tolerance);
void allocateDeviceMemory(void** ptr, size_t size);
void freeDeviceMemory(void* ptr);
cudaError_t transferDataToDevice(void* dst, const void* src, size_t size);
cudaError_t transferDataToHost(void* dst, const void* src, size_t size);
void computeStatistics(const float* data, size_t n, float* mean, float* stddev);

#endif // HEADER_H
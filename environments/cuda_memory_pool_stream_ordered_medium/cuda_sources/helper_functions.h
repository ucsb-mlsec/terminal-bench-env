#ifndef HELPER_FUNCTIONS_H
#define HELPER_FUNCTIONS_H

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdint.h>

// Type definitions for kernel parameters
typedef struct {
    int width;
    int height;
    int channels;
} ImageDimensions;

typedef struct {
    float x;
    float y;
    float z;
} Vector3D;

// Inline helper function to check CUDA errors
inline void checkCudaError(cudaError_t error, const char* file, int line) {
    if (error != cudaSuccess) {
        fprintf(stderr, "CUDA Error at %s:%d: %s\n", file, line, 
                cudaGetErrorString(error));
    }
}

#define CUDA_CHECK(call) checkCudaError(call, __FILE__, __LINE__)

// Template function to calculate grid dimensions
template<typename T>
inline dim3 calculateGridDim(T totalElements, int blockSize) {
    int numBlocks = (totalElements + blockSize - 1) / blockSize;
    return dim3(numBlocks, 1, 1);
}

// Inline utility to compute aligned size
inline size_t getAlignedSize(size_t size, size_t alignment = 256) {
    return ((size + alignment - 1) / alignment) * alignment;
}

// Function prototype for device property queries
void printDeviceProperties(int deviceId);

// Function prototype for timing utilities
double getElapsedTime(cudaEvent_t start, cudaEvent_t stop);

// Inline function to validate dimensions
inline bool validateDimensions(const ImageDimensions* dims) {
    return (dims->width > 0 && dims->height > 0 && dims->channels > 0);
}

#endif // HELPER_FUNCTIONS_H
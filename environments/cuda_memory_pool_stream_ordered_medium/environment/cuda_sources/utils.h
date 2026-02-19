#ifndef CUDA_UTILS_H
#define CUDA_UTILS_H

#include <stdio.h>
#include <stdint.h>

// Constants for CUDA kernel configuration
#define MAX_THREADS_PER_BLOCK 1024
#define WARP_SIZE 32
#define MAX_GRID_SIZE 65535

// Error checking macro
#define CHECK_CUDA_ERROR(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// Structure for kernel launch configuration
typedef struct {
    dim3 blockSize;
    dim3 gridSize;
    size_t sharedMemBytes;
} KernelConfig;

// Inline utility function to calculate grid dimensions
inline int calculateGridSize(int totalElements, int blockSize) {
    return (totalElements + blockSize - 1) / blockSize;
}

// Function declarations
void printDeviceProperties(int deviceId);
int getOptimalBlockSize(size_t dataSize);
double calculateBandwidth(size_t bytes, float timeMs);

#endif // CUDA_UTILS_H
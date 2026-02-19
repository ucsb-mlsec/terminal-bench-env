#ifndef CUDA_H
#define CUDA_H

// Mock CUDA header file for build system compatibility
// This is not a functional CUDA implementation

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// Error types
typedef enum cudaError {
    cudaSuccess = 0,
    cudaErrorMemoryAllocation = 2,
    cudaErrorInitializationError = 3,
    cudaErrorInvalidValue = 11,
    cudaErrorInvalidDevicePointer = 17,
    cudaErrorInvalidMemcpyDirection = 21
} cudaError_t;

// Memory copy types
typedef enum cudaMemcpyKind {
    cudaMemcpyHostToHost = 0,
    cudaMemcpyHostToDevice = 1,
    cudaMemcpyDeviceToHost = 2,
    cudaMemcpyDeviceToDevice = 3
} cudaMemcpyKind_t;

// Stream type
typedef struct CUstream_st* cudaStream_t;

// Device properties structure
typedef struct cudaDeviceProp {
    char name[256];
    size_t totalGlobalMem;
    int major;
    int minor;
} cudaDeviceProp_t;

// Mock function declarations
cudaError_t cudaMalloc(void** devPtr, size_t size);
cudaError_t cudaFree(void* devPtr);
cudaError_t cudaMemcpy(void* dst, const void* src, size_t count, cudaMemcpyKind_t kind);
cudaError_t cudaMemset(void* devPtr, int value, size_t count);
cudaError_t cudaDeviceSynchronize(void);
cudaError_t cudaGetDeviceCount(int* count);
cudaError_t cudaGetDeviceProperties(cudaDeviceProp_t* prop, int device);
cudaError_t cudaSetDevice(int device);
const char* cudaGetErrorString(cudaError_t error);

#ifdef __cplusplus
}
#endif

#endif // CUDA_H
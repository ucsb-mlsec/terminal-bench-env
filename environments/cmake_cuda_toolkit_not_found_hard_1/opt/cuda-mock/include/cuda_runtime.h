#ifndef CUDA_RUNTIME_H
#define CUDA_RUNTIME_H

/* Mock CUDA Runtime API Header */
/* This is a stub implementation for build system compatibility */

#include "cuda.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Device properties structure */
typedef struct cudaDeviceProp {
    char name[256];
    size_t totalGlobalMem;
    size_t sharedMemPerBlock;
    int regsPerBlock;
    int warpSize;
    int maxThreadsPerBlock;
    int maxThreadsDim[3];
    int maxGridSize[3];
    int clockRate;
    int major;
    int minor;
} cudaDeviceProp;

/* Runtime API function declarations */
cudaError_t cudaDeviceSynchronize(void);
cudaError_t cudaGetDeviceCount(int* count);
cudaError_t cudaGetDeviceProperties(cudaDeviceProp* prop, int device);
cudaError_t cudaSetDevice(int device);
cudaError_t cudaGetDevice(int* device);
cudaError_t cudaMalloc(void** devPtr, size_t size);
cudaError_t cudaFree(void* devPtr);
cudaError_t cudaMemcpy(void* dst, const void* src, size_t count, enum cudaMemcpyKind kind);
cudaError_t cudaGetLastError(void);
const char* cudaGetErrorString(cudaError_t error);

/* Memory copy kinds */
enum cudaMemcpyKind {
    cudaMemcpyHostToHost = 0,
    cudaMemcpyHostToDevice = 1,
    cudaMemcpyDeviceToHost = 2,
    cudaMemcpyDeviceToDevice = 3
};

#ifdef __cplusplus
}
#endif

#endif /* CUDA_RUNTIME_H */
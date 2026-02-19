#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error in %s:%d: %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            return err; \
        } \
    } while(0)

cudaError_t allocateDeviceMemory(void** d_ptr, size_t size) {
    if (d_ptr == NULL || size == 0) {
        fprintf(stderr, "Invalid parameters for memory allocation\n");
        return cudaErrorInvalidValue;
    }
    
    CUDA_CHECK(cudaMalloc(d_ptr, size));
    printf("Allocated %zu bytes on device at %p\n", size, *d_ptr);
    return cudaSuccess;
}

cudaError_t freeDeviceMemory(void* d_ptr) {
    if (d_ptr == NULL) {
        fprintf(stderr, "Attempting to free NULL pointer\n");
        return cudaErrorInvalidValue;
    }
    
    CUDA_CHECK(cudaFree(d_ptr));
    printf("Freed device memory at %p\n", d_ptr);
    return cudaSuccess;
}

cudaError_t copyToDevice(void* d_dst, const void* h_src, size_t size) {
    if (d_dst == NULL || h_src == NULL || size == 0) {
        fprintf(stderr, "Invalid parameters for host to device copy\n");
        return cudaErrorInvalidValue;
    }
    
    CUDA_CHECK(cudaMemcpy(d_dst, h_src, size, cudaMemcpyHostToDevice));
    printf("Copied %zu bytes from host to device\n", size);
    return cudaSuccess;
}

cudaError_t copyFromDevice(void* h_dst, const void* d_src, size_t size) {
    if (h_dst == NULL || d_src == NULL || size == 0) {
        fprintf(stderr, "Invalid parameters for device to host copy\n");
        return cudaErrorInvalidValue;
    }
    
    CUDA_CHECK(cudaMemcpy(h_dst, d_src, size, cudaMemcpyDeviceToHost));
    printf("Copied %zu bytes from device to host\n", size);
    return cudaSuccess;
}

cudaError_t copyDeviceToDevice(void* d_dst, const void* d_src, size_t size) {
    if (d_dst == NULL || d_src == NULL || size == 0) {
        fprintf(stderr, "Invalid parameters for device to device copy\n");
        return cudaErrorInvalidValue;
    }
    
    CUDA_CHECK(cudaMemcpy(d_dst, d_src, size, cudaMemcpyDeviceToDevice));
    return cudaSuccess;
}

cudaError_t setDeviceMemory(void* d_ptr, int value, size_t size) {
    if (d_ptr == NULL || size == 0) {
        return cudaErrorInvalidValue;
    }
    
    CUDA_CHECK(cudaMemset(d_ptr, value, size));
    return cudaSuccess;
}
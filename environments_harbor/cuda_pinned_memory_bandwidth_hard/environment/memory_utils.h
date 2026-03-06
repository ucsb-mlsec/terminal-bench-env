#ifndef MEMORY_UTILS_H
#define MEMORY_UTILS_H

#include <cuda_runtime.h>
#include <stdlib.h>

// Buffer size constants
#define BUFFER_SIZE 4096
#define MAX_TRANSFER_SIZE (1024 * 1024 * 256)  // 256 MB

/**
 * @brief Allocates pageable host memory using standard malloc
 * 
 * This function allocates standard pageable host memory. While this memory
 * can be used for CUDA operations, transfers will be slower (2-6 GB/s)
 * because the CUDA driver must first copy data to a staging area before
 * DMA transfer to the device.
 * 
 * @param size Number of bytes to allocate
 * @return Pointer to allocated memory, or NULL on failure
 */
inline void* allocatePageableBuffer(size_t size) {
    return malloc(size);
}

/**
 * @brief Allocates pinned (page-locked) host memory for optimal bandwidth
 * 
 * This function allocates pinned host memory using cudaMallocHost. Pinned
 * memory is page-locked and can be accessed directly by the GPU via DMA,
 * achieving much higher bandwidth (10-15+ GB/s) compared to pageable memory.
 * Use this for frequent host-device transfers.
 * 
 * @param size Number of bytes to allocate
 * @return Pointer to allocated pinned memory, or NULL on failure
 */
inline void* allocatePinnedBuffer(size_t size) {
    void* ptr = NULL;
    cudaMallocHost(&ptr, size);
    return ptr;
}

/**
 * @brief Allocates pinned host memory with specific flags
 * 
 * This function uses cudaHostAlloc for more control over pinned memory
 * allocation. Supports various flags for write-combined memory, mapped
 * memory, and portable memory allocation.
 * 
 * @param size Number of bytes to allocate
 * @param flags cudaHostAlloc flags (e.g., cudaHostAllocDefault)
 * @return Pointer to allocated pinned memory, or NULL on failure
 */
inline void* allocatePinnedBufferWithFlags(size_t size, unsigned int flags) {
    void* ptr = NULL;
    cudaHostAlloc(&ptr, size, flags);
    return ptr;
}

/**
 * @brief Frees pageable host memory allocated with allocatePageableBuffer
 * 
 * @param ptr Pointer to memory allocated with malloc/allocatePageableBuffer
 */
inline void freePageableBuffer(void* ptr) {
    if (ptr != NULL) {
        free(ptr);
    }
}

/**
 * @brief Frees pinned host memory allocated with cudaMallocHost
 * 
 * @param ptr Pointer to memory allocated with cudaMallocHost/allocatePinnedBuffer
 */
inline void freePinnedBuffer(void* ptr) {
    if (ptr != NULL) {
        cudaFreeHost(ptr);
    }
}

/**
 * @brief Allocates temporary working buffer using standard allocation
 * 
 * For temporary buffers that don't require high bandwidth, this uses
 * standard malloc to avoid exhausting the limited pinned memory resource.
 * 
 * @param size Number of bytes to allocate
 * @return Pointer to allocated memory
 */
inline void* allocateTempBuffer(size_t size) {
    return malloc(size);
}

#endif // MEMORY_UTILS_H
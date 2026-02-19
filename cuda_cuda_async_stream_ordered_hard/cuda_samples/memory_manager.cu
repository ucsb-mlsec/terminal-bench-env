// memory_manager.cu
// Traditional CUDA Memory Management Wrapper
// Copyright (c) 2019-2021 Research Computing Team
// Implements classic CUDA memory allocation patterns with pooling support

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <map>
#include <vector>
#include <mutex>

// Error checking macro for CUDA calls
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error in %s:%d: %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

/**
 * DeviceMemoryManager - Traditional CUDA memory management wrapper
 * Provides allocation, deallocation, and copy operations using classic CUDA APIs
 * Includes optional memory pooling for improved performance
 */
class DeviceMemoryManager {
private:
    struct MemoryBlock {
        void* ptr;
        size_t size;
        bool in_use;
        
        MemoryBlock() : ptr(nullptr), size(0), in_use(false) {}
        MemoryBlock(void* p, size_t s) : ptr(p), size(s), in_use(true) {}
    };
    
    std::vector<MemoryBlock> memory_pool;
    std::map<void*, size_t> active_allocations;
    std::mutex pool_mutex;
    size_t total_allocated;
    size_t peak_usage;
    bool enable_pooling;
    
public:
    /**
     * Constructor
     * @param pooling Enable memory pooling to reuse freed blocks
     */
    DeviceMemoryManager(bool pooling = false) 
        : total_allocated(0), peak_usage(0), enable_pooling(pooling) {
        printf("DeviceMemoryManager initialized (pooling: %s)\n", 
               pooling ? "enabled" : "disabled");
    }
    
    /**
     * Destructor - frees all allocated device memory
     */
    ~DeviceMemoryManager() {
        cleanup();
        printf("DeviceMemoryManager destroyed. Peak usage: %zu bytes\n", peak_usage);
    }
    
    /**
     * Allocate device memory using traditional cudaMalloc
     * @param size Number of bytes to allocate
     * @return Pointer to allocated device memory
     */
    void* allocate(size_t size) {
        std::lock_guard<std::mutex> lock(pool_mutex);
        
        if (size == 0) {
            fprintf(stderr, "Warning: Attempted to allocate 0 bytes\n");
            return nullptr;
        }
        
        // Try to reuse from pool if pooling is enabled
        if (enable_pooling) {
            for (auto& block : memory_pool) {
                if (!block.in_use && block.size >= size) {
                    block.in_use = true;
                    active_allocations[block.ptr] = block.size;
                    printf("Reused pooled memory: %zu bytes at %p\n", block.size, block.ptr);
                    return block.ptr;
                }
            }
        }
        
        // Allocate new memory using traditional cudaMalloc
        void* device_ptr = nullptr;
        cudaError_t err = cudaMalloc(&device_ptr, size);
        
        if (err != cudaSuccess) {
            fprintf(stderr, "cudaMalloc failed for %zu bytes: %s\n", 
                    size, cudaGetErrorString(err));
            return nullptr;
        }
        
        // Track allocation
        active_allocations[device_ptr] = size;
        total_allocated += size;
        
        if (total_allocated > peak_usage) {
            peak_usage = total_allocated;
        }
        
        if (enable_pooling) {
            memory_pool.push_back(MemoryBlock(device_ptr, size));
        }
        
        printf("Allocated %zu bytes at %p (total: %zu bytes)\n", 
               size, device_ptr, total_allocated);
        
        return device_ptr;
    }
    
    /**
     * Deallocate device memory
     * @param ptr Pointer to device memory to free
     */
    void deallocate(void* ptr) {
        std::lock_guard<std::mutex> lock(pool_mutex);
        
        if (ptr == nullptr) {
            return;
        }
        
        auto it = active_allocations.find(ptr);
        if (it == active_allocations.end()) {
            fprintf(stderr, "Warning: Attempted to free untracked pointer %p\n", ptr);
            return;
        }
        
        size_t size = it->second;
        total_allocated -= size;
        active_allocations.erase(it);
        
        if (enable_pooling) {
            // Mark as available in pool instead of actually freeing
            for (auto& block : memory_pool) {
                if (block.ptr == ptr) {
                    block.in_use = false;
                    printf("Returned %zu bytes at %p to pool\n", size, ptr);
                    return;
                }
            }
        }
        
        // Actually free the memory using traditional cudaFree
        CUDA_CHECK(cudaFree(ptr));
        printf("Freed %zu bytes at %p\n", size, ptr);
    }
    
    /**
     * Copy data from host to device
     * @param dst Device pointer (destination)
     * @param src Host pointer (source)
     * @param size Number of bytes to copy
     */
    void copyHostToDevice(void* dst, const void* src, size_t size) {
        if (dst == nullptr || src == nullptr || size == 0) {
            fprintf(stderr, "Invalid copy parameters\n");
            return;
        }
        
        CUDA_CHECK(cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice));
        printf("Copied %zu bytes from host to device\n", size);
    }
    
    /**
     * Copy data from device to host
     * @param dst Host pointer (destination)
     * @param src Device pointer (source)
     * @param size Number of bytes to copy
     */
    void copyDeviceToHost(void* dst, const void* src, size_t size) {
        if (dst == nullptr || src == nullptr || size == 0) {
            fprintf(stderr, "Invalid copy parameters\n");
            return;
        }
        
        CUDA_CHECK(cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost));
        printf("Copied %zu bytes from device to host\n", size);
    }
    
    /**
     * Copy data from device to device
     * @param dst Device pointer (destination)
     * @param src Device pointer (source)
     * @param size Number of bytes to copy
     */
    void copyDeviceToDevice(void* dst, const void* src, size_t size) {
        if (dst == nullptr || src == nullptr || size == 0) {
            fprintf(stderr, "Invalid copy parameters\n");
            return;
        }
        
        CUDA_CHECK(cudaMemcpy(dst, src, size, cudaMemcpyDeviceToDevice));
        printf("Copied %zu bytes device-to-device\n", size);
    }
    
    /**
     * Set device memory to a specific value
     * @param ptr Device pointer
     * @param value Byte value to set
     * @param size Number of bytes to set
     */
    void memset(void* ptr, int value, size_t size) {
        if (ptr == nullptr || size == 0) {
            return;
        }
        
        CUDA_CHECK(cudaMemset(ptr, value, size));
        printf("Memset %zu bytes at %p to value %d\n", size, ptr, value);
    }
    
    /**
     * Get current total allocated memory
     * @return Total bytes currently allocated
     */
    size_t getTotalAllocated() const {
        return total_allocated;
    }
    
    /**
     * Get peak memory usage
     * @return Peak bytes allocated
     */
    size_t getPeakUsage() const {
        return peak_usage;
    }
    
    /**
     * Get number of active allocations
     * @return Count of active allocations
     */
    size_t getActiveAllocationCount() const {
        return active_allocations.size();
    }
    
    /**
     * Clean up all memory - frees everything in pool and active allocations
     */
    void cleanup() {
        std::lock_guard<std::mutex> lock(pool_mutex);
        
        printf("Cleaning up memory manager...\n");
        
        // Free all blocks in pool
        for (auto& block : memory_pool) {
            if (block.ptr != nullptr) {
                cudaFree(block.ptr);
                printf("Cleaned up pooled block: %zu bytes at %p\n", 
                       block.size, block.ptr);
            }
        }
        
        memory_pool.clear();
        active_allocations.clear();
        total_allocated = 0;
        
        printf("Memory manager cleanup complete\n");
    }
    
    /**
     * Print current memory statistics
     */
    void printStats() const {
        printf("\n=== Memory Manager Statistics ===\n");
        printf("Total allocated: %zu bytes\n", total_allocated);
        printf("Peak usage: %zu bytes\n", peak_usage);
        printf("Active allocations: %zu\n", active_allocations.size());
        printf("Pool size: %zu blocks\n", memory_pool.size());
        printf("Pooling: %s\n", enable_pooling ? "enabled" : "disabled");
        printf("================================\n\n");
    }
};

// Example usage and test function
/*
void testMemoryManager() {
    printf("Testing DeviceMemoryManager...\n\n");
    
    // Create manager with pooling enabled
    DeviceMemoryManager manager(true);
    
    // Test allocation
    size_t size1 = 1024 * 1024; // 1 MB
    void* ptr1 = manager.allocate(size1);
    
    size_t size2 = 2048 * 1024; // 2 MB
    void* ptr2 = manager.allocate(size2);
    
    // Test host-to-device copy
    float* host_data = new float[256];
    for (int i = 0; i < 256; i++) {
        host_data[i] = static_cast<float>(i);
    }
    
    void* ptr3 = manager.allocate(256 * sizeof(float));
    manager.copyHostToDevice(ptr3, host_data, 256 * sizeof(float));
    
    // Test memset
    manager.memset(ptr3, 0, 256 * sizeof(float));
    
    // Print statistics
    manager.printStats();
    
    // Test deallocation
    manager.deallocate(ptr1);
    manager.deallocate(ptr2);
    
    // Test reallocation (should reuse from pool)
    void* ptr4 = manager.allocate(size1);
    
    // Copy device to host
    float* result_data = new float[256];
    manager.copyDeviceToHost(result_data, ptr3, 256 * sizeof(float));
    
    // Final statistics
    manager.printStats();
    
    // Cleanup
    manager.deallocate(ptr3);
    manager.deallocate(ptr4);
    delete[] host_data;
    delete[] result_data;
    
    printf("Test complete\n");
}

int main() {
    testMemoryManager();
    return 0;
}
*/
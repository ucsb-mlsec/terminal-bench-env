#include <cuda_runtime.h>
#include <stdio.h>

// Utility kernel for copying data between arrays
__global__ void copyData(int* src, int* dst, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        dst[idx] = src[idx];
    }
}

// Utility kernel for validating computation results
__global__ void validateResults(int* data, int* flags, int size, int threshold) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        if (data[idx] > threshold) {
            flags[idx] = 1;
            // Launch error check if validation fails
            if (idx == 0) {
                errorCheck<<<1, 32>>>(flags, size);
            }
        } else {
            flags[idx] = 0;
        }
    }
}

// Debug print kernel that outputs diagnostic information
__global__ void debugPrint(int* data, int size, int level) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx == 0) {
        printf("Debug Level %d: First element = %d\n", level, data[0]);
    }
    
    // Sync memory after debug output
    if (idx == 0 && level < 3) {
        memorySync<<<1, 32>>>(data, size);
    }
}

// Memory synchronization utility kernel
__global__ void memorySync(int* data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    __syncthreads();
    
    if (idx < size) {
        // Perform memory barrier operations
        int temp = data[idx];
        __threadfence();
        data[idx] = temp;
    }
    
    // Call copyData for backup if first thread
    if (idx == 0) {
        int* backup = data + size;
        copyData<<<(size + 255) / 256, 256>>>(data, backup, size);
    }
}

// Error checking kernel that validates flags and triggers debug output
__global__ void errorCheck(int* flags, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    __shared__ int errorCount;
    
    if (threadIdx.x == 0) {
        errorCount = 0;
    }
    __syncthreads();
    
    if (idx < size && flags[idx] != 0) {
        atomicAdd(&errorCount, 1);
    }
    __syncthreads();
    
    if (threadIdx.x == 0 && errorCount > 0) {
        printf("Errors detected: %d\n", errorCount);
        // Launch debug print to investigate
        debugPrint<<<1, 1>>>(flags, size, 1);
        
        // Also call processing kernel from another file
        processDataBlock<<<(size + 255) / 256, 256>>>(flags, size, 0);
    }
}
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define N 4096
#define BLOCK_SIZE 256
#define MATRIX_SIZE 1024
#define PAGE_SIZE 4096

// Kernel 1: Severely non-coalesced strided access pattern
// CRITICAL ISSUE: stride of 1024 causes threads in same warp to access different pages
__global__ void stridedAccessKernel(float *data, int stride, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < n) {
        // Non-coalesced access - threads access memory 1024 elements apart
        int idx = tid * stride;
        if (idx < n * stride) {
            data[idx] = data[idx] * 2.0f + 1.0f;
        }
    }
}

// Kernel 2: Scattered random access through indirection
// CRITICAL ISSUE: index array causes completely random memory access pattern
__global__ void scatteredAccessKernel(float *data, int *indices, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < n) {
        // Random access pattern through index array - causes page thrashing
        int idx = indices[tid];
        if (idx < n) {
            data[idx] = data[idx] * 3.0f;
        }
    }
}

// Kernel 3: Backward iteration causing reverse page faults
// ISSUE: Accessing memory in reverse order causes unnecessary page migrations
__global__ void reverseAccessKernel(float *input, float *output, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < n) {
        // Reverse access pattern
        int idx = n - 1 - tid;
        output[idx] = input[idx] + 5.0f;
    }
}

// Kernel 4: Column-major access on row-major data (matrix transpose-like)
// ISSUE: Non-coalesced memory access in matrix operations
__global__ void columnMajorAccessKernel(float *matrix, int rows, int cols) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < cols) {
        // Column-major iteration on row-major stored data
        for (int i = 0; i < rows; i++) {
            int idx = i * cols + tid;  // Accessing column-wise
            matrix[idx] = matrix[idx] * 1.5f;
        }
    }
}

// Kernel 5: Atomic operations on unified memory without hints
// ISSUE: Atomic ops on unified memory cause excessive coherency traffic
__global__ void atomicAccessKernel(int *counter, float *data, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < n) {
        float value = data[tid * 64];  // Also non-coalesced
        if (value > 0.5f) {
            atomicAdd(counter, 1);  // Atomic on unified memory
        }
    }
}

// Kernel 6: Reduction with poor memory access pattern
// ISSUE: Divergent memory access during reduction phases
__global__ void poorReductionKernel(float *data, float *result, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    
    // Each thread processes non-contiguous elements
    float sum = 0.0f;
    for (int i = tid; i < n; i += blockDim.x * gridDim.x * 8) {
        sum += data[i];
    }
    
    // Non-coalesced write to result
    if (tid < n / 8) {
        result[tid * 16] = sum;
    }
}

// Kernel 7: Nested loop with page-boundary crossing pattern
// ISSUE: Access pattern crosses page boundaries frequently
__global__ void pageBoundaryCrossKernel(float *data, int width, int height) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int x = tid % width;
    int y = tid / width;
    
    if (y < height && x < width) {
        // Access pattern that crosses page boundaries
        for (int dy = -2; dy <= 2; dy++) {
            for (int dx = -2; dx <= 2; dx++) {
                int nx = x + dx;
                int ny = y + dy;
                if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                    int idx = ny * width + nx;
                    data[tid] += data[idx] * 0.1f;  // Scattered access
                }
            }
        }
    }
}

// Host function with multiple thrashing patterns
void processDataWithThrashing() {
    // Allocate unified memory
    float *managed_data;
    float *managed_output;
    float *managed_matrix;
    int *managed_indices;
    int *managed_counter;
    float *managed_result;
    
    cudaMallocManaged(&managed_data, N * 1024 * sizeof(float));
    cudaMallocManaged(&managed_output, N * sizeof(float));
    cudaMallocManaged(&managed_matrix, MATRIX_SIZE * MATRIX_SIZE * sizeof(float));
    cudaMallocManaged(&managed_indices, N * sizeof(int));
    cudaMallocManaged(&managed_counter, sizeof(int));
    cudaMallocManaged(&managed_result, N * sizeof(float));
    
    // Initialize data on CPU
    for (int i = 0; i < N * 1024; i++) {
        managed_data[i] = (float)i * 0.01f;
    }
    
    for (int i = 0; i < N; i++) {
        managed_indices[i] = (i * 997) % N;  // Random-like pattern
    }
    
    // Initialize matrix
    for (int i = 0; i < MATRIX_SIZE * MATRIX_SIZE; i++) {
        managed_matrix[i] = 1.0f;
    }
    
    *managed_counter = 0;
    
    // CRITICAL ISSUE: Launch kernel immediately after CPU init without prefetch
    // Missing: cudaMemPrefetchAsync(managed_data, N * 1024 * sizeof(float), 0, 0);
    int blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    stridedAccessKernel<<<blocks, BLOCK_SIZE>>>(managed_data, 1024, N);
    
    // CRITICAL ISSUE: CPU accesses memory without synchronization while GPU might still be working
    // Missing: cudaDeviceSynchronize();
    float cpu_sum = 0.0f;
    for (int i = 0; i < 1000; i++) {
        cpu_sum += managed_data[i * 1024];  // CPU reading same strided pattern
    }
    
    // Launch scattered access kernel
    scatteredAccessKernel<<<blocks, BLOCK_SIZE>>>(managed_data, managed_indices, N);
    
    // ISSUE: CPU modifies data immediately after kernel launch without sync
    for (int i = 0; i < 500; i++) {
        managed_data[managed_indices[i]] *= 2.0f;
    }
    
    // Launch reverse access kernel
    reverseAccessKernel<<<blocks, BLOCK_SIZE>>>(managed_data, managed_output, N);
    
    // Launch column-major access kernel
    int matrix_blocks = (MATRIX_SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE;
    columnMajorAccessKernel<<<matrix_blocks, BLOCK_SIZE>>>(managed_matrix, MATRIX_SIZE, MATRIX_SIZE);
    
    // ISSUE: Alternating CPU-GPU access without synchronization
    cudaDeviceSynchronize();  // At least there's a sync here
    
    // CPU reads matrix data
    float matrix_sum = 0.0f;
    for (int i = 0; i < MATRIX_SIZE; i++) {
        for (int j = 0; j < MATRIX_SIZE; j++) {
            matrix_sum += managed_matrix[i * MATRIX_SIZE + j];
        }
    }
    
    // Launch atomic kernel - no prefetch back to GPU
    atomicAccessKernel<<<blocks, BLOCK_SIZE>>>(managed_counter, managed_data, N);
    
    // CRITICAL ISSUE: CPU reads counter without sync
    printf("Counter value: %d\n", *managed_counter);
    
    // Launch reduction kernel
    poorReductionKernel<<<blocks, BLOCK_SIZE>>>(managed_data, managed_result, N * 1024);
    
    // Launch page boundary kernel
    int page_blocks = (MATRIX_SIZE * MATRIX_SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE;
    pageBoundaryCrossKernel<<<page_blocks, BLOCK_SIZE>>>(managed_matrix, MATRIX_SIZE, MATRIX_SIZE);
    
    // ISSUE: Multiple kernel launches without intermediate synchronization
    // causes overlapping CPU-GPU access potential
    
    cudaDeviceSynchronize();
    
    // ISSUE: CPU iterates through data in page-sized chunks - thrashing pattern
    for (int i = 0; i < N * 1024; i += PAGE_SIZE / sizeof(float)) {
        managed_data[i] = managed_data[i] + managed_result[i % N];
    }
    
    // Launch another kernel without prefetch
    stridedAccessKernel<<<blocks, BLOCK_SIZE>>>(managed_data, 2048, N);
    
    // ISSUE: Immediate CPU access after kernel launch
    float final_value = managed_data[0];
    printf("Final value: %f\n", final_value);
    
    cudaDeviceSynchronize();
    
    printf("CPU sum: %f, Matrix sum: %f\n", cpu_sum, matrix_sum);
    
    // Cleanup
    cudaFree(managed_data);
    cudaFree(managed_output);
    cudaFree(managed_matrix);
    cudaFree(managed_indices);
    cudaFree(managed_counter);
    cudaFree(managed_result);
}

// Additional function demonstrating simultaneous access pattern
void simultaneousAccessPattern() {
    float *shared_buffer;
    cudaMallocManaged(&shared_buffer, N * 512 * sizeof(float));
    
    // Initialize on CPU
    for (int i = 0; i < N * 512; i++) {
        shared_buffer[i] = (float)i;
    }
    
    // Launch long-running kernel
    int blocks = (N * 512 + BLOCK_SIZE - 1) / BLOCK_SIZE;
    pageBoundaryCrossKernel<<<blocks, BLOCK_SIZE>>>(shared_buffer, N, 512);
    
    // CRITICAL ISSUE: CPU accesses same buffer while GPU kernel is running
    // No synchronization between launch and CPU access
    for (int i = 0; i < N * 512; i += 128) {
        shared_buffer[i] = shared_buffer[i] * 0.5f;
    }
    
    cudaDeviceSynchronize();
    cudaFree(shared_buffer);
}

int main() {
    printf("Starting CUDA Unified Memory Test\n");
    
    // Test basic patterns
    processDataWithThrashing();
    
    // Test simultaneous access
    simultaneousAccessPattern();
    
    printf("Test completed\n");
    return 0;
}
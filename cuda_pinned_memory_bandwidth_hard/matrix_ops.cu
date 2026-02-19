#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define MATRIX_SIZE 1024
#define BLOCK_SIZE 16

// Macro for checking CUDA errors
#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(error)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// Matrix multiplication kernel
__global__ void matrixMultiplyKernel(float *A, float *B, float *C, int width) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < width && col < width) {
        float sum = 0.0f;
        for (int k = 0; k < width; k++) {
            sum += A[row * width + k] * B[k * width + col];
        }
        C[row * width + col] = sum;
    }
}

// Matrix transpose kernel
__global__ void matrixTransposeKernel(float *input, float *output, int width) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < width && col < width) {
        output[col * width + row] = input[row * width + col];
    }
}

// Initialize matrix with random values
void initializeMatrix(float *matrix, int size) {
    for (int i = 0; i < size; i++) {
        matrix[i] = (float)(rand() % 100) / 10.0f;
    }
}

// Print a portion of the matrix for verification
void printMatrixSample(float *matrix, int width, int sampleSize) {
    printf("Matrix sample (%dx%d):\n", sampleSize, sampleSize);
    for (int i = 0; i < sampleSize && i < width; i++) {
        for (int j = 0; j < sampleSize && j < width; j++) {
            printf("%.2f ", matrix[i * width + j]);
        }
        printf("\n");
    }
}

int main() {
    int matrixElements = MATRIX_SIZE * MATRIX_SIZE;
    size_t matrixBytes = matrixElements * sizeof(float);
    
    // Host matrices - using PINNED memory for optimal bandwidth
    // Pinned memory allows Direct Memory Access (DMA) for much higher transfer speeds
    // Typical bandwidth: 10-15+ GB/s vs 2-6 GB/s for pageable memory
    float *h_A, *h_B, *h_C, *h_temp;
    
    printf("Allocating pinned host memory for high-bandwidth transfers...\n");
    
    // Allocate pinned memory for matrix A using cudaMallocHost
    CUDA_CHECK(cudaMallocHost((void**)&h_A, matrixBytes));
    
    // Allocate pinned memory for matrix B using cudaMallocHost
    CUDA_CHECK(cudaMallocHost((void**)&h_B, matrixBytes));
    
    // Allocate pinned memory for result matrix C using cudaMallocHost
    CUDA_CHECK(cudaMallocHost((void**)&h_C, matrixBytes));
    
    // Allocate pinned memory for temporary matrix using cudaHostAlloc
    CUDA_CHECK(cudaHostAlloc((void**)&h_temp, matrixBytes, cudaHostAllocDefault));
    
    printf("Pinned memory allocated successfully\n");
    
    // Device matrices
    float *d_A, *d_B, *d_C, *d_temp;
    
    printf("Allocating device memory...\n");
    CUDA_CHECK(cudaMalloc((void**)&d_A, matrixBytes));
    CUDA_CHECK(cudaMalloc((void**)&d_B, matrixBytes));
    CUDA_CHECK(cudaMalloc((void**)&d_C, matrixBytes));
    CUDA_CHECK(cudaMalloc((void**)&d_temp, matrixBytes));
    
    // Initialize host matrices
    printf("Initializing matrices...\n");
    initializeMatrix(h_A, matrixElements);
    initializeMatrix(h_B, matrixElements);
    
    // Setup kernel execution configuration
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim((MATRIX_SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE,
                 (MATRIX_SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE);
    
    // Create CUDA events for timing
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    // Record start time
    CUDA_CHECK(cudaEventRecord(start));
    
    // Transfer matrices from host to device
    // High bandwidth transfer thanks to pinned memory
    printf("Transferring data to device...\n");
    CUDA_CHECK(cudaMemcpy(d_A, h_A, matrixBytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, matrixBytes, cudaMemcpyHostToDevice));
    
    // Launch matrix multiplication kernel
    printf("Launching matrix multiplication kernel...\n");
    matrixMultiplyKernel<<<gridDim, blockDim>>>(d_A, d_B, d_C, MATRIX_SIZE);
    CUDA_CHECK(cudaGetLastError());
    
    // Launch matrix transpose kernel on result
    printf("Launching matrix transpose kernel...\n");
    matrixTransposeKernel<<<gridDim, blockDim>>>(d_C, d_temp, MATRIX_SIZE);
    CUDA_CHECK(cudaGetLastError());
    
    // Wait for kernels to complete
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Transfer result back to host
    // High bandwidth transfer thanks to pinned memory
    printf("Transferring results back to host...\n");
    CUDA_CHECK(cudaMemcpy(h_C, d_C, matrixBytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_temp, d_temp, matrixBytes, cudaMemcpyDeviceToHost));
    
    // Record stop time
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    // Calculate elapsed time
    float milliseconds = 0;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    printf("Total execution time: %.3f ms\n", milliseconds);
    
    // Print sample of results
    printf("\nOriginal result matrix:\n");
    printMatrixSample(h_C, MATRIX_SIZE, 4);
    
    printf("\nTransposed result matrix:\n");
    printMatrixSample(h_temp, MATRIX_SIZE, 4);
    
    // Cleanup - free pinned host memory using cudaFreeHost
    printf("\nFreeing pinned host memory...\n");
    CUDA_CHECK(cudaFreeHost(h_A));
    CUDA_CHECK(cudaFreeHost(h_B));
    CUDA_CHECK(cudaFreeHost(h_C));
    CUDA_CHECK(cudaFreeHost(h_temp));
    
    // Free device memory
    printf("Freeing device memory...\n");
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    CUDA_CHECK(cudaFree(d_temp));
    
    // Destroy events
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    
    printf("Matrix operations completed successfully!\n");
    printf("Note: Pinned memory enabled optimal bandwidth for all transfers\n");
    
    return 0;
}
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define N 1048576  // 1M elements
#define BLOCK_SIZE 256
#define MATRIX_DIM 1024

// Kernel 1: Vector initialization and preprocessing
__global__ void preprocessVector(float *input, float *output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        // Apply preprocessing transformation
        output[idx] = sqrtf(fabsf(input[idx])) + 0.5f;
    }
}

// Kernel 2: Main computation - element-wise operations
__global__ void computeIntensiveOperation(float *a, float *b, float *result, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float temp = a[idx] * b[idx];
        // Simulate intensive computation
        for (int i = 0; i < 10; i++) {
            temp = temp * 0.99f + sinf(temp) * 0.01f;
        }
        result[idx] = temp;
    }
}

// Kernel 3: Parallel reduction for summation
__global__ void reduceSum(float *input, float *output, int size) {
    __shared__ float sdata[BLOCK_SIZE];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Load data into shared memory
    sdata[tid] = (idx < size) ? input[idx] : 0.0f;
    __syncthreads();
    
    // Perform reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    // Write result for this block
    if (tid == 0) {
        output[blockIdx.x] = sdata[0];
    }
}

// Kernel 4: Post-processing and normalization
__global__ void postprocessResults(float *data, float normFactor, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] = data[idx] / normFactor;
        // Apply final transformation
        data[idx] = tanhf(data[idx]);
    }
}

// Host function to check CUDA errors
void checkCudaError(const char *msg) {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error at %s: %s\n", msg, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

// Host function to initialize data
void initializeHostData(float *data, int size) {
    for (int i = 0; i < size; i++) {
        data[i] = (float)(rand() % 1000) / 100.0f - 5.0f;
    }
}

// Main computational pipeline
int main(int argc, char **argv) {
    printf("Starting Scientific Computing Pipeline\n");
    printf("Processing %d elements\n", N);
    
    // Host memory allocation
    float *h_input = (float*)malloc(N * sizeof(float));
    float *h_vectorA = (float*)malloc(N * sizeof(float));
    float *h_vectorB = (float*)malloc(N * sizeof(float));
    float *h_result = (float*)malloc(N * sizeof(float));
    float *h_reduced = (float*)malloc((N / BLOCK_SIZE) * sizeof(float));
    
    if (!h_input || !h_vectorA || !h_vectorB || !h_result || !h_reduced) {
        fprintf(stderr, "Host memory allocation failed\n");
        return EXIT_FAILURE;
    }
    
    // Initialize input data on host
    printf("Initializing host data...\n");
    initializeHostData(h_input, N);
    initializeHostData(h_vectorA, N);
    initializeHostData(h_vectorB, N);
    
    // Device memory pointers
    float *d_input, *d_preprocessed, *d_vectorA, *d_vectorB, *d_result, *d_reduced;
    
    // Allocate device memory
    printf("Allocating device memory...\n");
    cudaMalloc((void**)&d_input, N * sizeof(float));
    checkCudaError("cudaMalloc d_input");
    
    cudaMalloc((void**)&d_preprocessed, N * sizeof(float));
    checkCudaError("cudaMalloc d_preprocessed");
    
    cudaMalloc((void**)&d_vectorA, N * sizeof(float));
    checkCudaError("cudaMalloc d_vectorA");
    
    cudaMalloc((void**)&d_vectorB, N * sizeof(float));
    checkCudaError("cudaMalloc d_vectorB");
    
    cudaMalloc((void**)&d_result, N * sizeof(float));
    checkCudaError("cudaMalloc d_result");
    
    cudaMalloc((void**)&d_reduced, (N / BLOCK_SIZE) * sizeof(float));
    checkCudaError("cudaMalloc d_reduced");
    
    // Transfer input data from host to device
    printf("Copying data to device...\n");
    cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice);
    checkCudaError("cudaMemcpy H2D d_input");
    
    cudaMemcpy(d_vectorA, h_vectorA, N * sizeof(float), cudaMemcpyHostToDevice);
    checkCudaError("cudaMemcpy H2D d_vectorA");
    
    cudaMemcpy(d_vectorB, h_vectorB, N * sizeof(float), cudaMemcpyHostToDevice);
    checkCudaError("cudaMemcpy H2D d_vectorB");
    
    // Configure kernel launch parameters
    int numBlocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 blockDim(BLOCK_SIZE);
    dim3 gridDim(numBlocks);
    
    // Phase 1: Preprocessing
    printf("Launching preprocessing kernel...\n");
    preprocessVector<<<gridDim, blockDim>>>(d_input, d_preprocessed, N);
    cudaDeviceSynchronize();
    checkCudaError("preprocessVector kernel");
    
    // Phase 2: Main computation
    printf("Launching main computation kernel...\n");
    computeIntensiveOperation<<<gridDim, blockDim>>>(d_vectorA, d_vectorB, d_result, N);
    cudaDeviceSynchronize();
    checkCudaError("computeIntensiveOperation kernel");
    
    // Phase 3: Reduction operation
    printf("Launching reduction kernel...\n");
    int reduceBlocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    reduceSum<<<reduceBlocks, BLOCK_SIZE>>>(d_result, d_reduced, N);
    cudaDeviceSynchronize();
    checkCudaError("reduceSum kernel");
    
    // Copy reduction results back to host for final sum calculation
    cudaMemcpy(h_reduced, d_reduced, reduceBlocks * sizeof(float), cudaMemcpyDeviceToHost);
    checkCudaError("cudaMemcpy D2H d_reduced");
    
    // Final reduction on CPU
    float totalSum = 0.0f;
    for (int i = 0; i < reduceBlocks; i++) {
        totalSum += h_reduced[i];
    }
    float normFactor = totalSum / N;
    printf("Computed normalization factor: %f\n", normFactor);
    
    // Phase 4: Post-processing
    printf("Launching post-processing kernel...\n");
    postprocessResults<<<gridDim, blockDim>>>(d_result, normFactor, N);
    cudaDeviceSynchronize();
    checkCudaError("postprocessResults kernel");
    
    // Transfer final results back to host
    printf("Copying results back to host...\n");
    cudaMemcpy(h_result, d_result, N * sizeof(float), cudaMemcpyDeviceToHost);
    checkCudaError("cudaMemcpy D2H d_result");
    
    // Verify results (sample check)
    printf("Verifying results...\n");
    int sampleIndices[] = {0, N/4, N/2, 3*N/4, N-1};
    for (int i = 0; i < 5; i++) {
        int idx = sampleIndices[i];
        printf("Result[%d] = %f\n", idx, h_result[idx]);
    }
    
    // Cleanup device memory
    printf("Freeing device memory...\n");
    cudaFree(d_input);
    cudaFree(d_preprocessed);
    cudaFree(d_vectorA);
    cudaFree(d_vectorB);
    cudaFree(d_result);
    cudaFree(d_reduced);
    checkCudaError("cudaFree");
    
    // Cleanup host memory
    printf("Freeing host memory...\n");
    free(h_input);
    free(h_vectorA);
    free(h_vectorB);
    free(h_result);
    free(h_reduced);
    
    printf("Pipeline completed successfully!\n");
    
    return EXIT_SUCCESS;
}
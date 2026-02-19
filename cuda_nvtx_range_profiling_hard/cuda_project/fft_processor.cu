#include <cuda_runtime.h>
#include <cufft.h>
#include <nvToolsExt.h>
#include <stdio.h>
#include <stdlib.h>

#define SIGNAL_SIZE 1024
#define BATCH_SIZE 8

// Error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

#define CUFFT_CHECK(call) \
    do { \
        cufftResult err = call; \
        if (err != CUFFT_SUCCESS) { \
            fprintf(stderr, "cuFFT error at %s:%d: %d\n", __FILE__, __LINE__, err); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// Kernel for preprocessing signal data
__global__ void preprocessSignal(cufftComplex* data, int size, float scale) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx].x *= scale;
        data[idx].y *= scale;
    }
}

// Kernel for applying window function
__global__ void applyHammingWindow(cufftComplex* data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float window = 0.54f - 0.46f * cosf(2.0f * M_PI * idx / (size - 1));
        data[idx].x *= window;
        data[idx].y *= window;
    }
}

// Function to perform forward FFT on batch of signals
void performForwardFFT(cufftComplex* d_signal, cufftComplex* d_frequency, 
                       int signal_size, int batch_size) {
    nvtxRangePush("FFT Forward");
    
    cufftHandle plan;
    CUFFT_CHECK(cufftPlan1d(&plan, signal_size, CUFFT_C2C, batch_size));
    
    CUFFT_CHECK(cufftExecC2C(plan, d_signal, d_frequency, CUFFT_FORWARD));
    
    CUDA_CHECK(cudaDeviceSynchronize());
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA kernel error: %s\n", cudaGetErrorString(err));
    }
    
    cufftDestroy(plan);
    nvtxRangePop();
}

// Function to perform inverse FFT
void performInverseFFT(cufftComplex* d_frequency, cufftComplex* d_signal,
                       int signal_size, int batch_size) {
    nvtxRangePush("FFT Inverse");
    
    cufftHandle plan;
    CUFFT_CHECK(cufftPlan1d(&plan, signal_size, CUFFT_C2C, batch_size));
    
    CUFFT_CHECK(cufftExecC2C(plan, d_frequency, d_signal, CUFFT_INVERSE));
    
    CUDA_CHECK(cudaDeviceSynchronize());
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA kernel error: %s\n", cudaGetErrorString(err));
    }
    
    // Normalize after inverse FFT
    int threads = 256;
    int blocks = (signal_size * batch_size + threads - 1) / threads;
    float scale = 1.0f / signal_size;
    preprocessSignal<<<blocks, threads>>>(d_signal, signal_size * batch_size, scale);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    cufftDestroy(plan);
    nvtxRangePop();
}

// Function to preprocess data before FFT
void preprocessData(cufftComplex* d_data, int signal_size, int batch_size) {
    nvtxRangePush("Data Preprocessing");
    
    int total_size = signal_size * batch_size;
    int threads = 256;
    int blocks = (total_size + threads - 1) / threads;
    
    applyHammingWindow<<<blocks, threads>>>(d_data, total_size);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Preprocessing kernel error: %s\n", cudaGetErrorString(err));
    }
    
    nvtxRangePop();
}

// Main processing pipeline
int processSignalBatch() {
    int total_size = SIGNAL_SIZE * BATCH_SIZE;
    size_t data_bytes = total_size * sizeof(cufftComplex);
    
    // Allocate host memory
    cufftComplex* h_input = (cufftComplex*)malloc(data_bytes);
    cufftComplex* h_output = (cufftComplex*)malloc(data_bytes);
    
    // Initialize input data with sample signal
    for (int i = 0; i < total_size; i++) {
        h_input[i].x = sinf(2.0f * M_PI * 50.0f * i / SIGNAL_SIZE);
        h_input[i].y = 0.0f;
    }
    
    // Allocate device memory
    cufftComplex* d_signal;
    cufftComplex* d_frequency;
    CUDA_CHECK(cudaMalloc(&d_signal, data_bytes));
    CUDA_CHECK(cudaMalloc(&d_frequency, data_bytes));
    
    // Copy input to device
    CUDA_CHECK(cudaMemcpy(d_signal, h_input, data_bytes, cudaMemcpyHostToDevice));
    
    // Process the signal
    preprocessData(d_signal, SIGNAL_SIZE, BATCH_SIZE);
    performForwardFFT(d_signal, d_frequency, SIGNAL_SIZE, BATCH_SIZE);
    performInverseFFT(d_frequency, d_signal, SIGNAL_SIZE, BATCH_SIZE);
    
    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_output, d_signal, data_bytes, cudaMemcpyDeviceToHost));
    
    // Cleanup
    CUDA_CHECK(cudaFree(d_signal));
    CUDA_CHECK(cudaFree(d_frequency));
    free(h_input);
    free(h_output);
    
    return 0;
}
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define CHECK_CUDA(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(error)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

#define HISTOGRAM_BINS 256
#define BLOCK_SIZE 256

// Kernel for histogram calculation
__global__ void histogramKernel(const unsigned char* input, unsigned int* histogram, 
                                int width, int height) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_pixels = width * height;
    
    if (idx < total_pixels) {
        unsigned char value = input[idx];
        atomicAdd(&histogram[value], 1);
    }
}

// Kernel for histogram equalization
__global__ void equalizeKernel(const unsigned char* input, unsigned char* output,
                               const float* cdf, int width, int height) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_pixels = width * height;
    
    if (idx < total_pixels) {
        unsigned char pixel_value = input[idx];
        float normalized = cdf[pixel_value];
        output[idx] = (unsigned char)(normalized * 255.0f);
    }
}

// Kernel to compute cumulative distribution function
__global__ void computeCDFKernel(const unsigned int* histogram, float* cdf,
                                 int total_pixels, int num_bins) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < num_bins) {
        unsigned int cumulative = 0;
        for (int i = 0; i <= idx; i++) {
            cumulative += histogram[i];
        }
        cdf[idx] = (float)cumulative / (float)total_pixels;
    }
}

// Kernel for contrast adjustment
__global__ void contrastAdjustKernel(const unsigned char* input, unsigned char* output,
                                     float alpha, int beta, int width, int height) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_pixels = width * height;
    
    if (idx < total_pixels) {
        float value = (float)input[idx];
        float adjusted = alpha * value + beta;
        adjusted = fminf(255.0f, fmaxf(0.0f, adjusted));
        output[idx] = (unsigned char)adjusted;
    }
}

// Host function to perform histogram equalization
void performHistogramEqualization(unsigned char* h_input, unsigned char* h_output,
                                  int width, int height, cudaStream_t stream) {
    int total_pixels = width * height;
    size_t image_size = total_pixels * sizeof(unsigned char);
    
    // Allocate device memory using traditional cudaMalloc
    unsigned char* d_input;
    unsigned char* d_output;
    unsigned int* d_histogram;
    float* d_cdf;
    
    CHECK_CUDA(cudaMalloc(&d_input, image_size));
    CHECK_CUDA(cudaMalloc(&d_output, image_size));
    CHECK_CUDA(cudaMalloc(&d_histogram, HISTOGRAM_BINS * sizeof(unsigned int)));
    CHECK_CUDA(cudaMalloc(&d_cdf, HISTOGRAM_BINS * sizeof(float)));
    
    // Initialize histogram to zero
    CHECK_CUDA(cudaMemset(d_histogram, 0, HISTOGRAM_BINS * sizeof(unsigned int)));
    
    // Copy input data to device
    CHECK_CUDA(cudaMemcpy(d_input, h_input, image_size, cudaMemcpyHostToDevice));
    
    // Launch histogram kernel
    int grid_size = (total_pixels + BLOCK_SIZE - 1) / BLOCK_SIZE;
    histogramKernel<<<grid_size, BLOCK_SIZE, 0, stream>>>(d_input, d_histogram, width, height);
    CHECK_CUDA(cudaGetLastError());
    
    // Launch CDF computation kernel
    int cdf_grid_size = (HISTOGRAM_BINS + BLOCK_SIZE - 1) / BLOCK_SIZE;
    computeCDFKernel<<<cdf_grid_size, BLOCK_SIZE, 0, stream>>>(d_histogram, d_cdf, 
                                                                 total_pixels, HISTOGRAM_BINS);
    CHECK_CUDA(cudaGetLastError());
    
    // Launch equalization kernel
    equalizeKernel<<<grid_size, BLOCK_SIZE, 0, stream>>>(d_input, d_output, d_cdf, 
                                                          width, height);
    CHECK_CUDA(cudaGetLastError());
    
    // Copy result back to host
    CHECK_CUDA(cudaMemcpy(h_output, d_output, image_size, cudaMemcpyDeviceToHost));
    
    // Free device memory using traditional cudaFree
    CHECK_CUDA(cudaFree(d_input));
    CHECK_CUDA(cudaFree(d_output));
    CHECK_CUDA(cudaFree(d_histogram));
    CHECK_CUDA(cudaFree(d_cdf));
}

// Host function to perform contrast adjustment
void performContrastAdjustment(unsigned char* h_input, unsigned char* h_output,
                               int width, int height, float alpha, int beta) {
    int total_pixels = width * height;
    size_t image_size = total_pixels * sizeof(unsigned char);
    
    // Allocate device memory using traditional approach
    unsigned char* d_input;
    unsigned char* d_output;
    
    CHECK_CUDA(cudaMalloc(&d_input, image_size));
    CHECK_CUDA(cudaMalloc(&d_output, image_size));
    
    // Copy input to device
    CHECK_CUDA(cudaMemcpy(d_input, h_input, image_size, cudaMemcpyHostToDevice));
    
    // Launch contrast adjustment kernel
    int grid_size = (total_pixels + BLOCK_SIZE - 1) / BLOCK_SIZE;
    contrastAdjustKernel<<<grid_size, BLOCK_SIZE>>>(d_input, d_output, alpha, beta, 
                                                     width, height);
    CHECK_CUDA(cudaGetLastError());
    
    // Synchronize to ensure kernel completion
    CHECK_CUDA(cudaDeviceSynchronize());
    
    // Copy result back
    CHECK_CUDA(cudaMemcpy(h_output, d_output, image_size, cudaMemcpyDeviceToHost));
    
    // Free device memory
    CHECK_CUDA(cudaFree(d_input));
    CHECK_CUDA(cudaFree(d_output));
}

int main(int argc, char** argv) {
    printf("CUDA Histogram Equalization and Contrast Adjustment\n");
    printf("===================================================\n");
    
    // Image parameters
    int width = 1920;
    int height = 1080;
    int total_pixels = width * height;
    size_t image_size = total_pixels * sizeof(unsigned char);
    
    // Allocate host memory
    unsigned char* h_input = (unsigned char*)malloc(image_size);
    unsigned char* h_output_eq = (unsigned char*)malloc(image_size);
    unsigned char* h_output_contrast = (unsigned char*)malloc(image_size);
    
    if (!h_input || !h_output_eq || !h_output_contrast) {
        fprintf(stderr, "Failed to allocate host memory\n");
        return EXIT_FAILURE;
    }
    
    // Initialize input with synthetic data
    for (int i = 0; i < total_pixels; i++) {
        h_input[i] = (unsigned char)((i * 137) % 256);
    }
    
    // Create CUDA stream
    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));
    
    printf("Processing %dx%d image (%d pixels)\n", width, height, total_pixels);
    
    // Perform histogram equalization
    printf("Performing histogram equalization...\n");
    performHistogramEqualization(h_input, h_output_eq, width, height, stream);
    CHECK_CUDA(cudaStreamSynchronize(stream));
    printf("Histogram equalization complete\n");
    
    // Perform contrast adjustment
    printf("Performing contrast adjustment...\n");
    float alpha = 1.5f;
    int beta = 10;
    performContrastAdjustment(h_input, h_output_contrast, width, height, alpha, beta);
    printf("Contrast adjustment complete\n");
    
    // Verify results
    unsigned long long sum_eq = 0;
    unsigned long long sum_contrast = 0;
    for (int i = 0; i < total_pixels; i++) {
        sum_eq += h_output_eq[i];
        sum_contrast += h_output_contrast[i];
    }
    
    printf("\nResults:\n");
    printf("  Equalized average: %.2f\n", (float)sum_eq / total_pixels);
    printf("  Contrast adjusted average: %.2f\n", (float)sum_contrast / total_pixels);
    
    // Cleanup
    CHECK_CUDA(cudaStreamDestroy(stream));
    free(h_input);
    free(h_output_eq);
    free(h_output_contrast);
    
    printf("\nProcessing completed successfully\n");
    
    return EXIT_SUCCESS;
}
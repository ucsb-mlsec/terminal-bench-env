#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define CHECK_CUDA_ERROR(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error in %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

#define SOBEL_KERNEL_SIZE 3
#define BLOCK_SIZE 16

// Sobel edge detection kernel
__global__ void sobelEdgeDetectionKernel(const unsigned char* input, 
                                          unsigned char* output,
                                          int width, 
                                          int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    // Sobel operators for edge detection
    int Gx[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
    int Gy[3][3] = {{-1, -2, -1}, {0, 0, 0}, {1, 2, 1}};
    
    int sumX = 0;
    int sumY = 0;
    
    // Apply convolution
    for (int ky = -1; ky <= 1; ky++) {
        for (int kx = -1; kx <= 1; kx++) {
            int pixelX = min(max(x + kx, 0), width - 1);
            int pixelY = min(max(y + ky, 0), height - 1);
            int pixelValue = input[pixelY * width + pixelX];
            
            sumX += pixelValue * Gx[ky + 1][kx + 1];
            sumY += pixelValue * Gy[ky + 1][kx + 1];
        }
    }
    
    // Calculate gradient magnitude
    int magnitude = static_cast<int>(sqrt(static_cast<float>(sumX * sumX + sumY * sumY)));
    magnitude = min(max(magnitude, 0), 255);
    
    output[y * width + x] = static_cast<unsigned char>(magnitude);
}

// Initialize test image with synthetic pattern
void initializeTestImage(unsigned char* image, int width, int height) {
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            // Create a test pattern with gradients and edges
            int value = (x + y) % 256;
            if ((x / 50) % 2 == 0) {
                value = 255 - value;
            }
            if (y > height / 2 && y < height / 2 + 50) {
                value = 128;
            }
            image[y * width + x] = static_cast<unsigned char>(value);
        }
    }
}

// Validate output by checking for edge values
void validateOutput(const unsigned char* output, int width, int height) {
    int edgePixelCount = 0;
    long long sumValues = 0;
    
    for (int i = 0; i < width * height; i++) {
        if (output[i] > 50) {
            edgePixelCount++;
        }
        sumValues += output[i];
    }
    
    double avgIntensity = static_cast<double>(sumValues) / (width * height);
    printf("Edge detection complete:\n");
    printf("  - Edge pixels (>50): %d (%.2f%%)\n", 
           edgePixelCount, 100.0 * edgePixelCount / (width * height));
    printf("  - Average intensity: %.2f\n", avgIntensity);
}

int main(int argc, char** argv) {
    printf("=== CUDA Sobel Edge Detection (Traditional Memory Management) ===\n\n");
    
    // Image dimensions
    const int width = 1920;
    const int height = 1080;
    const size_t imageSize = width * height * sizeof(unsigned char);
    
    printf("Processing image: %dx%d pixels (%.2f MB)\n", 
           width, height, imageSize / (1024.0 * 1024.0));
    
    // Allocate host memory using traditional approach
    unsigned char* h_input = (unsigned char*)malloc(imageSize);
    unsigned char* h_output = (unsigned char*)malloc(imageSize);
    
    if (!h_input || !h_output) {
        fprintf(stderr, "Failed to allocate host memory\n");
        return EXIT_FAILURE;
    }
    
    // Initialize input image
    printf("Initializing test image...\n");
    initializeTestImage(h_input, width, height);
    
    // Allocate device memory using traditional cudaMalloc
    unsigned char* d_input = nullptr;
    unsigned char* d_output = nullptr;
    
    printf("Allocating device memory...\n");
    CHECK_CUDA_ERROR(cudaMalloc(&d_input, imageSize));
    CHECK_CUDA_ERROR(cudaMalloc(&d_output, imageSize));
    
    // Copy input data to device using standard cudaMemcpy
    printf("Transferring data to device...\n");
    CHECK_CUDA_ERROR(cudaMemcpy(d_input, h_input, imageSize, cudaMemcpyHostToDevice));
    
    // Configure kernel launch parameters
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim((width + BLOCK_SIZE - 1) / BLOCK_SIZE,
                 (height + BLOCK_SIZE - 1) / BLOCK_SIZE);
    
    printf("Launching Sobel edge detection kernel...\n");
    printf("  Grid: (%d, %d), Block: (%d, %d)\n", 
           gridDim.x, gridDim.y, blockDim.x, blockDim.y);
    
    // Create CUDA events for timing
    cudaEvent_t start, stop;
    CHECK_CUDA_ERROR(cudaEventCreate(&start));
    CHECK_CUDA_ERROR(cudaEventCreate(&stop));
    
    // Record start time
    CHECK_CUDA_ERROR(cudaEventRecord(start));
    
    // Launch kernel on default stream (no stream parameter)
    sobelEdgeDetectionKernel<<<gridDim, blockDim>>>(d_input, d_output, width, height);
    
    // Record stop time
    CHECK_CUDA_ERROR(cudaEventRecord(stop));
    
    // Synchronize device using traditional approach
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    
    // Calculate elapsed time
    float milliseconds = 0;
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&milliseconds, start, stop));
    printf("Kernel execution time: %.3f ms\n", milliseconds);
    
    // Copy results back to host using standard cudaMemcpy
    printf("Transferring results to host...\n");
    CHECK_CUDA_ERROR(cudaMemcpy(h_output, d_output, imageSize, cudaMemcpyDeviceToHost));
    
    // Validate output
    validateOutput(h_output, width, height);
    
    // Clean up device memory using traditional cudaFree
    printf("Cleaning up device memory...\n");
    CHECK_CUDA_ERROR(cudaFree(d_input));
    CHECK_CUDA_ERROR(cudaFree(d_output));
    
    // Clean up events
    CHECK_CUDA_ERROR(cudaEventDestroy(start));
    CHECK_CUDA_ERROR(cudaEventDestroy(stop));
    
    // Free host memory
    free(h_input);
    free(h_output);
    
    printf("\nProcessing complete. Traditional memory management used throughout.\n");
    
    return EXIT_SUCCESS;
}
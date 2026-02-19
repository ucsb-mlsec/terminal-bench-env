#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

// Image dimensions
#define WIDTH 1920
#define HEIGHT 1080
#define CHANNELS 3

// CUDA kernel for converting RGB image to grayscale
__global__ void rgb_to_grayscale_kernel(unsigned char *input, unsigned char *output, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        int idx = (y * width + x) * CHANNELS;
        int gray_idx = y * width + x;
        
        // Standard grayscale conversion formula
        unsigned char r = input[idx];
        unsigned char g = input[idx + 1];
        unsigned char b = input[idx + 2];
        
        output[gray_idx] = (unsigned char)(0.299f * r + 0.587f * g + 0.114f * b);
    }
}

// CUDA kernel for applying a simple blur filter
__global__ void blur_kernel(unsigned char *input, unsigned char *output, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x > 0 && x < width - 1 && y > 0 && y < height - 1) {
        int idx = y * width + x;
        
        // Simple 3x3 average blur
        int sum = 0;
        for (int dy = -1; dy <= 1; dy++) {
            for (int dx = -1; dx <= 1; dx++) {
                sum += input[(y + dy) * width + (x + dx)];
            }
        }
        output[idx] = sum / 9;
    }
}

// Function to check CUDA errors
void checkCudaError(cudaError_t error, const char *message) {
    if (error != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s - %s\n", message, cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }
}

// Main image processing function
int process_image() {
    printf("Starting image processing pipeline...\n");
    printf("Image dimensions: %dx%d\n", WIDTH, HEIGHT);
    
    size_t rgb_size = WIDTH * HEIGHT * CHANNELS * sizeof(unsigned char);
    size_t gray_size = WIDTH * HEIGHT * sizeof(unsigned char);
    
    // ============================================================================
    // HOST MEMORY ALLOCATION STRATEGY: Using standard pageable memory (malloc)
    // ============================================================================
    // This approach uses standard C malloc() which allocates pageable memory.
    // Pageable memory results in lower bandwidth (typically 2-6 GB/s) because:
    // 1. The memory pages can be swapped to disk by the OS
    // 2. CUDA driver must copy data to a staging area before DMA transfer
    // 3. Cannot use Direct Memory Access (DMA) directly from pageable memory
    // ============================================================================
    
    // Allocate host memory for input RGB image using malloc (PAGEABLE memory)
    printf("Allocating host memory for input image using malloc (pageable)...\n");
    unsigned char *h_input = (unsigned char*)malloc(rgb_size);
    if (h_input == NULL) {
        fprintf(stderr, "Failed to allocate host input memory\n");
        return -1;
    }
    
    // Allocate host memory for intermediate grayscale image using malloc (PAGEABLE memory)
    printf("Allocating host memory for grayscale image using malloc (pageable)...\n");
    unsigned char *h_grayscale = (unsigned char*)malloc(gray_size);
    if (h_grayscale == NULL) {
        fprintf(stderr, "Failed to allocate host grayscale memory\n");
        free(h_input);
        return -1;
    }
    
    // Initialize input image with sample data
    printf("Initializing input image with sample data...\n");
    for (int i = 0; i < WIDTH * HEIGHT * CHANNELS; i++) {
        h_input[i] = (unsigned char)(i % 256);
    }
    
    // ============================================================================
    // DEVICE MEMORY ALLOCATION (these should be ignored by the analyzer)
    // ============================================================================
    
    unsigned char *d_input, *d_grayscale, *d_output;
    
    // Allocate device memory for input RGB image
    printf("Allocating device memory...\n");
    cudaError_t error = cudaMalloc((void**)&d_input, rgb_size);
    checkCudaError(error, "Failed to allocate device input memory");
    
    // Allocate device memory for grayscale image
    error = cudaMalloc((void**)&d_grayscale, gray_size);
    checkCudaError(error, "Failed to allocate device grayscale memory");
    
    // Allocate device memory for output image
    error = cudaMalloc((void**)&d_output, gray_size);
    checkCudaError(error, "Failed to allocate device output memory");
    
    // ============================================================================
    // MEMORY TRANSFER: Host to Device
    // ============================================================================
    // Due to using pageable memory (malloc), this transfer will be slower
    // The CUDA driver must first copy data to a temporary pinned buffer
    // before initiating DMA transfer to the device
    // ============================================================================
    
    printf("Copying input data from host to device...\n");
    error = cudaMemcpy(d_input, h_input, rgb_size, cudaMemcpyHostToDevice);
    checkCudaError(error, "Failed to copy input data to device");
    
    // ============================================================================
    // KERNEL EXECUTION
    // ============================================================================
    
    // Configure kernel launch parameters
    dim3 block_size(16, 16);
    dim3 grid_size((WIDTH + block_size.x - 1) / block_size.x,
                   (HEIGHT + block_size.y - 1) / block_size.y);
    
    // Launch RGB to grayscale conversion kernel
    printf("Launching RGB to grayscale kernel...\n");
    rgb_to_grayscale_kernel<<<grid_size, block_size>>>(d_input, d_grayscale, WIDTH, HEIGHT);
    error = cudaGetLastError();
    checkCudaError(error, "Kernel launch failed");
    
    // Launch blur kernel
    printf("Launching blur kernel...\n");
    blur_kernel<<<grid_size, block_size>>>(d_grayscale, d_output, WIDTH, HEIGHT);
    error = cudaGetLastError();
    checkCudaError(error, "Kernel launch failed");
    
    // Wait for kernels to complete
    error = cudaDeviceSynchronize();
    checkCudaError(error, "Device synchronization failed");
    
    // ============================================================================
    // MEMORY TRANSFER: Device to Host
    // ============================================================================
    // Again, due to pageable memory, this transfer suffers from reduced bandwidth
    // ============================================================================
    
    printf("Copying results from device to host...\n");
    error = cudaMemcpy(h_grayscale, d_output, gray_size, cudaMemcpyDeviceToHost);
    checkCudaError(error, "Failed to copy results from device");
    
    // Simple verification
    printf("First few output pixels: %d %d %d %d\n", 
           h_grayscale[0], h_grayscale[1], h_grayscale[2], h_grayscale[3]);
    
    // ============================================================================
    // CLEANUP: Free all allocated memory
    // ============================================================================
    
    printf("Cleaning up memory...\n");
    
    // Free host memory (pageable memory allocated with malloc)
    free(h_input);
    free(h_grayscale);
    
    // Free device memory
    cudaFree(d_input);
    cudaFree(d_grayscale);
    cudaFree(d_output);
    
    printf("Image processing completed successfully!\n");
    return 0;
}

// Main entry point
int main() {
    printf("========================================\n");
    printf("CUDA Image Processing Application\n");
    printf("Memory Strategy: Pageable (Suboptimal)\n");
    printf("========================================\n\n");
    
    int result = process_image();
    
    if (result == 0) {
        printf("\nSUCCESS: Processing completed\n");
    } else {
        printf("\nERROR: Processing failed\n");
    }
    
    return result;
}
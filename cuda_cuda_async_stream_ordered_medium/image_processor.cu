#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

// Image processing CUDA application with stream-ordered memory management
// Production code for high-performance image filter pipeline

#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(error)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// Image dimensions
#define WIDTH 1920
#define HEIGHT 1080
#define CHANNELS 3
#define IMAGE_SIZE (WIDTH * HEIGHT * CHANNELS)

// Kernel to apply brightness adjustment
__global__ void brightnessKernel(unsigned char* input, unsigned char* output, int size, float factor) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        int val = (int)(input[idx] * factor);
        output[idx] = (unsigned char)(val > 255 ? 255 : val);
    }
}

// Kernel to apply blur filter
__global__ void blurKernel(unsigned char* input, unsigned char* output, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    int idx = (y * width + x) * CHANNELS;
    
    // Simple 3x3 average blur
    for (int c = 0; c < CHANNELS; c++) {
        int sum = 0;
        int count = 0;
        for (int dy = -1; dy <= 1; dy++) {
            for (int dx = -1; dx <= 1; dx++) {
                int nx = x + dx;
                int ny = y + dy;
                if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                    sum += input[(ny * width + nx) * CHANNELS + c];
                    count++;
                }
            }
        }
        output[idx + c] = (unsigned char)(sum / count);
    }
}

// Kernel to apply contrast adjustment
__global__ void contrastKernel(unsigned char* input, unsigned char* output, int size, float factor) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float val = ((input[idx] / 255.0f - 0.5f) * factor + 0.5f) * 255.0f;
        output[idx] = (unsigned char)(val < 0 ? 0 : (val > 255 ? 255 : val));
    }
}

// Main image processing pipeline
int main(int argc, char** argv) {
    printf("Starting multi-stream image processor...\n");
    
    // Create multiple streams for concurrent processing
    cudaStream_t stream1, stream2, stream3;
    CUDA_CHECK(cudaStreamCreate(&stream1));
    CUDA_CHECK(cudaStreamCreate(&stream2));
    CUDA_CHECK(cudaStreamCreate(&stream3));
    
    // Create events for synchronization
    cudaEvent_t event1, event2;
    CUDA_CHECK(cudaEventCreate(&event1));
    CUDA_CHECK(cudaEventCreate(&event2));
    
    // Allocate host memory for input image
    unsigned char* h_input = (unsigned char*)malloc(IMAGE_SIZE);
    unsigned char* h_output = (unsigned char*)malloc(IMAGE_SIZE);
    
    // Initialize input image with sample data
    for (int i = 0; i < IMAGE_SIZE; i++) {
        h_input[i] = (unsigned char)(i % 256);
    }
    
    // Stream-ordered allocations for device memory
    unsigned char *d_input, *d_temp1, *d_temp2, *d_output;
    
    // Allocate input buffer on stream1
    CUDA_CHECK(cudaMallocAsync(&d_input, IMAGE_SIZE, stream1));
    
    // Allocate temporary buffer on stream2
    CUDA_CHECK(cudaMallocAsync(&d_temp1, IMAGE_SIZE, stream2));
    
    // Allocate second temporary buffer on stream3
    CUDA_CHECK(cudaMallocAsync(&d_temp2, IMAGE_SIZE, stream3));
    
    // Allocate output buffer on stream1
    CUDA_CHECK(cudaMallocAsync(&d_output, IMAGE_SIZE, stream1));
    
    // Copy input data to device on stream1
    CUDA_CHECK(cudaMemcpyAsync(d_input, h_input, IMAGE_SIZE, cudaMemcpyHostToDevice, stream1));
    
    // Configure kernel launch parameters
    int threadsPerBlock = 256;
    int blocksPerGrid = (IMAGE_SIZE + threadsPerBlock - 1) / threadsPerBlock;
    
    dim3 blockDim2D(16, 16);
    dim3 gridDim2D((WIDTH + blockDim2D.x - 1) / blockDim2D.x,
                   (HEIGHT + blockDim2D.y - 1) / blockDim2D.y);
    
    // VIOLATION 1: Launch brightness kernel on stream2 using d_input allocated on stream1
    // without waiting for stream1 to complete allocation and copy
    brightnessKernel<<<blocksPerGrid, threadsPerBlock, 0, stream2>>>(d_input, d_temp1, IMAGE_SIZE, 1.2f);
    
    // Record event on stream2 after brightness processing
    CUDA_CHECK(cudaEventRecord(event1, stream2));
    
    // VIOLATION 2: Launch blur kernel on stream3 using d_temp1 which was written by stream2
    // Missing synchronization - should wait for event1 before accessing d_temp1
    blurKernel<<<gridDim2D, blockDim2D, 0, stream3>>>(d_temp1, d_temp2, WIDTH, HEIGHT);
    
    // Properly synchronize stream1 to wait for stream3
    CUDA_CHECK(cudaEventRecord(event2, stream3));
    CUDA_CHECK(cudaStreamWaitEvent(stream1, event2, 0));
    
    // Launch contrast kernel on stream1 using d_temp2
    contrastKernel<<<blocksPerGrid, threadsPerBlock, 0, stream1>>>(d_temp2, d_output, IMAGE_SIZE, 1.5f);
    
    // Copy result back to host on stream1
    CUDA_CHECK(cudaMemcpyAsync(h_output, d_output, IMAGE_SIZE, cudaMemcpyDeviceToHost, stream1));
    
    // VIOLATION 3: Free d_temp1 on stream1, but it was allocated on stream2
    // and last used on stream3 without proper synchronization
    CUDA_CHECK(cudaFreeAsync(d_temp1, stream1));
    
    // Free d_input on stream2 (this is where it was originally used, but still needs sync)
    CUDA_CHECK(cudaFreeAsync(d_input, stream2));
    
    // VIOLATION 4: Free d_temp2 on stream2, but it was allocated on stream3
    // and last written by stream3, then read by stream1
    CUDA_CHECK(cudaFreeAsync(d_temp2, stream2));
    
    // Free d_output on stream1 (correct - same stream as allocation)
    CUDA_CHECK(cudaFreeAsync(d_output, stream1));
    
    // Synchronize all streams before cleanup
    CUDA_CHECK(cudaStreamSynchronize(stream1));
    CUDA_CHECK(cudaStreamSynchronize(stream2));
    CUDA_CHECK(cudaStreamSynchronize(stream3));
    
    // Verify output (simple check)
    printf("Processing complete. First pixel value: %d\n", h_output[0]);
    printf("Last pixel value: %d\n", h_output[IMAGE_SIZE - 1]);
    
    // Cleanup
    free(h_input);
    free(h_output);
    
    CUDA_CHECK(cudaEventDestroy(event1));
    CUDA_CHECK(cudaEventDestroy(event2));
    
    CUDA_CHECK(cudaStreamDestroy(stream1));
    CUDA_CHECK(cudaStreamDestroy(stream2));
    CUDA_CHECK(cudaStreamDestroy(stream3));
    
    printf("Image processing pipeline completed successfully.\n");
    
    return 0;
}
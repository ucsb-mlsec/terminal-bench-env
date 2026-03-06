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

// Kernel for brightness adjustment on grayscale images
__global__ void adjustBrightnessKernel(const unsigned char* input, 
                                       unsigned char* output,
                                       int width, 
                                       int height,
                                       float brightness_factor) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        int idx = y * width + x;
        float pixel_value = input[idx] * brightness_factor;
        
        // Clamp to valid range [0, 255]
        if (pixel_value > 255.0f) pixel_value = 255.0f;
        if (pixel_value < 0.0f) pixel_value = 0.0f;
        
        output[idx] = (unsigned char)pixel_value;
    }
}

// Kernel for applying Gaussian blur
__global__ void gaussianBlurKernel(const unsigned char* input,
                                   unsigned char* output,
                                   int width,
                                   int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        // Simple 3x3 Gaussian kernel
        float kernel[3][3] = {
            {1.0f/16.0f, 2.0f/16.0f, 1.0f/16.0f},
            {2.0f/16.0f, 4.0f/16.0f, 2.0f/16.0f},
            {1.0f/16.0f, 2.0f/16.0f, 1.0f/16.0f}
        };
        
        float sum = 0.0f;
        
        for (int dy = -1; dy <= 1; dy++) {
            for (int dx = -1; dx <= 1; dx++) {
                int nx = x + dx;
                int ny = y + dy;
                
                // Handle boundary conditions
                if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                    int idx = ny * width + nx;
                    sum += input[idx] * kernel[dy + 1][dx + 1];
                }
            }
        }
        
        output[y * width + x] = (unsigned char)sum;
    }
}

// Function to initialize test image data
void initializeImageData(unsigned char* data, int width, int height) {
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            // Create a gradient pattern
            data[y * width + x] = (unsigned char)((x + y) % 256);
        }
    }
}

// Main processing pipeline using stream-ordered allocation
int main(int argc, char** argv) {
    const int width = 1920;
    const int height = 1080;
    const size_t image_size = width * height * sizeof(unsigned char);
    
    printf("CUDA Image Processing Pipeline with Stream-Ordered Allocation\n");
    printf("Image dimensions: %dx%d\n", width, height);
    printf("Image size: %zu bytes\n", image_size);
    
    // Create CUDA stream for asynchronous operations
    cudaStream_t stream;
    CHECK_CUDA_ERROR(cudaStreamCreate(&stream));
    
    // Allocate host memory
    unsigned char* h_input = (unsigned char*)malloc(image_size);
    unsigned char* h_output = (unsigned char*)malloc(image_size);
    
    if (!h_input || !h_output) {
        fprintf(stderr, "Failed to allocate host memory\n");
        return EXIT_FAILURE;
    }
    
    // Initialize input image with test pattern
    initializeImageData(h_input, width, height);
    printf("Input image initialized\n");
    
    // Device pointers
    unsigned char* d_input = nullptr;
    unsigned char* d_temp = nullptr;
    unsigned char* d_output = nullptr;
    
    // Allocate device memory using stream-ordered allocation
    CHECK_CUDA_ERROR(cudaMallocAsync(&d_input, image_size, stream));
    CHECK_CUDA_ERROR(cudaMallocAsync(&d_temp, image_size, stream));
    CHECK_CUDA_ERROR(cudaMallocAsync(&d_output, image_size, stream));
    
    printf("Device memory allocated using cudaMallocAsync\n");
    
    // Copy input data to device asynchronously
    CHECK_CUDA_ERROR(cudaMemcpyAsync(d_input, h_input, image_size, 
                                     cudaMemcpyHostToDevice, stream));
    
    // Configure kernel launch parameters
    dim3 block_dim(16, 16);
    dim3 grid_dim((width + block_dim.x - 1) / block_dim.x,
                  (height + block_dim.y - 1) / block_dim.y);
    
    printf("Launching kernels with grid(%d, %d) and block(%d, %d)\n",
           grid_dim.x, grid_dim.y, block_dim.x, block_dim.y);
    
    // First processing stage: brightness adjustment
    float brightness_factor = 1.2f;
    adjustBrightnessKernel<<<grid_dim, block_dim, 0, stream>>>(
        d_input, d_temp, width, height, brightness_factor);
    
    CHECK_CUDA_ERROR(cudaGetLastError());
    
    // Second processing stage: Gaussian blur
    gaussianBlurKernel<<<grid_dim, block_dim, 0, stream>>>(
        d_temp, d_output, width, height);
    
    CHECK_CUDA_ERROR(cudaGetLastError());
    
    // Copy result back to host asynchronously
    CHECK_CUDA_ERROR(cudaMemcpyAsync(h_output, d_output, image_size,
                                     cudaMemcpyDeviceToHost, stream));
    
    // Synchronize stream to ensure all operations complete
    CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));
    
    printf("Processing completed successfully\n");
    
    // Verify output (simple checksum)
    unsigned long long checksum = 0;
    for (int i = 0; i < width * height; i++) {
        checksum += h_output[i];
    }
    printf("Output checksum: %llu\n", checksum);
    
    // Free device memory using stream-ordered deallocation
    CHECK_CUDA_ERROR(cudaFreeAsync(d_input, stream));
    CHECK_CUDA_ERROR(cudaFreeAsync(d_temp, stream));
    CHECK_CUDA_ERROR(cudaFreeAsync(d_output, stream));
    
    printf("Device memory freed using cudaFreeAsync\n");
    
    // Synchronize before destroying stream
    CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));
    
    // Destroy stream
    CHECK_CUDA_ERROR(cudaStreamDestroy(stream));
    
    // Free host memory
    free(h_input);
    free(h_output);
    
    printf("Pipeline execution completed\n");
    
    return EXIT_SUCCESS;
}
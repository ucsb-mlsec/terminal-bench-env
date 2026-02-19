#include <cuda_runtime.h>
#include <nvToolsExt.h>
#include <stdio.h>

#define MASK_WIDTH 5
#define TILE_WIDTH 16

// Constant memory for convolution mask
__constant__ float d_Mask[MASK_WIDTH * MASK_WIDTH];

// 2D Convolution Kernel
__global__ void convolution2DKernel(float *input, float *output, 
                                     int width, int height, int maskWidth)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (col < width && row < height) {
        float result = 0.0f;
        int maskRadius = maskWidth / 2;
        
        for (int maskRow = 0; maskRow < maskWidth; maskRow++) {
            for (int maskCol = 0; maskCol < maskWidth; maskCol++) {
                int imageRow = row - maskRadius + maskRow;
                int imageCol = col - maskRadius + maskCol;
                
                if (imageRow >= 0 && imageRow < height && 
                    imageCol >= 0 && imageCol < width) {
                    float imageValue = input[imageRow * width + imageCol];
                    float maskValue = d_Mask[maskRow * maskWidth + maskCol];
                    result += imageValue * maskValue;
                }
            }
        }
        
        output[row * width + col] = result;
    }
}

// Host function to perform convolution
void performConvolution(float *h_input, float *h_output, float *h_mask,
                        int width, int height, int maskWidth)
{
    float *d_input, *d_output;
    size_t imageSize = width * height * sizeof(float);
    size_t maskSize = maskWidth * maskWidth * sizeof(float);
    
    // Allocate device memory
    cudaMalloc((void**)&d_input, imageSize);
    cudaMalloc((void**)&d_output, imageSize);
    
    // Copy input data to device
    cudaMemcpy(d_input, h_input, imageSize, cudaMemcpyHostToDevice);
    
    // Copy mask to constant memory
    cudaMemcpyToSymbol(d_Mask, h_mask, maskSize);
    
    // Setup execution configuration
    dim3 blockDim(TILE_WIDTH, TILE_WIDTH);
    dim3 gridDim((width + blockDim.x - 1) / blockDim.x,
                 (height + blockDim.y - 1) / blockDim.y);
    
    // Launch kernel with NVTX instrumentation
    nvtxRangePush("2D Convolution");
    convolution2DKernel<<<gridDim, blockDim>>>(d_input, d_output, 
                                                width, height, maskWidth);
    cudaDeviceSynchronize();
    nvtxRangePop();
    
    // Copy result back to host
    cudaMemcpy(h_output, d_output, imageSize, cudaMemcpyDeviceToHost);
    
    // Free device memory
    cudaFree(d_input);
    cudaFree(d_output);
}

// Initialize a Gaussian mask
void initializeGaussianMask(float *mask, int maskWidth)
{
    float sigma = 1.0f;
    float sum = 0.0f;
    int radius = maskWidth / 2;
    
    for (int i = 0; i < maskWidth; i++) {
        for (int j = 0; j < maskWidth; j++) {
            int x = i - radius;
            int y = j - radius;
            float value = expf(-(x*x + y*y) / (2.0f * sigma * sigma));
            mask[i * maskWidth + j] = value;
            sum += value;
        }
    }
    
    // Normalize mask
    for (int i = 0; i < maskWidth * maskWidth; i++) {
        mask[i] /= sum;
    }
}

int main()
{
    int width = 1024;
    int height = 1024;
    int maskWidth = MASK_WIDTH;
    
    size_t imageSize = width * height * sizeof(float);
    size_t maskSize = maskWidth * maskWidth * sizeof(float);
    
    float *h_input = (float*)malloc(imageSize);
    float *h_output = (float*)malloc(imageSize);
    float *h_mask = (float*)malloc(maskSize);
    
    // Initialize input image
    for (int i = 0; i < width * height; i++) {
        h_input[i] = (float)(rand() % 256);
    }
    
    // Initialize Gaussian mask
    initializeGaussianMask(h_mask, maskWidth);
    
    // Perform convolution
    performConvolution(h_input, h_output, h_mask, width, height, maskWidth);
    
    printf("Convolution completed successfully\n");
    
    free(h_input);
    free(h_output);
    free(h_mask);
    
    return 0;
}
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdexcept>

// Error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(error)); \
            throw std::runtime_error(cudaGetErrorString(error)); \
        } \
    } while(0)

// Kernel for data normalization
__global__ void normalizeKernel(float* data, int size, float minVal, float maxVal) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float range = maxVal - minVal;
        if (range > 0.0f) {
            data[idx] = (data[idx] - minVal) / range;
        }
    }
}

// Kernel for type conversion from unsigned char to float
__global__ void convertUInt8ToFloatKernel(const unsigned char* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = static_cast<float>(input[idx]) / 255.0f;
    }
}

// Kernel for type conversion from float to unsigned char
__global__ void convertFloatToUInt8Kernel(const float* input, unsigned char* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float val = input[idx] * 255.0f;
        val = fminf(fmaxf(val, 0.0f), 255.0f);
        output[idx] = static_cast<unsigned char>(val);
    }
}

// Kernel for boundary padding with reflection
__global__ void reflectBoundaryKernel(const float* input, float* output, 
                                      int width, int height, int padSize) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    int outWidth = width + 2 * padSize;
    int outHeight = height + 2 * padSize;
    
    if (x < outWidth && y < outHeight) {
        int srcX = x - padSize;
        int srcY = y - padSize;
        
        if (srcX < 0) srcX = -srcX;
        if (srcX >= width) srcX = 2 * width - srcX - 2;
        if (srcY < 0) srcY = -srcY;
        if (srcY >= height) srcY = 2 * height - srcY - 2;
        
        srcX = min(max(srcX, 0), width - 1);
        srcY = min(max(srcY, 0), height - 1);
        
        output[y * outWidth + x] = input[srcY * width + srcX];
    }
}

// Stream-ordered utility function for computing min/max values
void computeMinMaxAsync(const float* d_data, int size, float* h_minVal, float* h_maxVal, 
                        cudaStream_t stream) {
    // Allocate temporary reduction buffers using stream-ordered allocation
    float* d_tempMin;
    float* d_tempMax;
    
    CUDA_CHECK(cudaMallocAsync(&d_tempMin, sizeof(float), stream));
    CUDA_CHECK(cudaMallocAsync(&d_tempMax, sizeof(float), stream));
    
    // Initialize with first element
    CUDA_CHECK(cudaMemcpyAsync(d_tempMin, d_data, sizeof(float), 
                               cudaMemcpyDeviceToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_tempMax, d_data, sizeof(float), 
                               cudaMemcpyDeviceToDevice, stream));
    
    // Simple reduction (in production, use a proper reduction algorithm)
    // This is simplified for demonstration
    CUDA_CHECK(cudaMemcpyAsync(h_minVal, d_tempMin, sizeof(float), 
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(h_maxVal, d_tempMax, sizeof(float), 
                               cudaMemcpyDeviceToHost, stream));
    
    // Free using stream-ordered deallocation
    CUDA_CHECK(cudaFreeAsync(d_tempMin, stream));
    CUDA_CHECK(cudaFreeAsync(d_tempMax, stream));
}

// Stream-ordered utility function for image normalization with temporary workspace
void normalizeImageAsync(float* d_data, int size, cudaStream_t stream) {
    // Allocate temporary buffers for min/max using stream-ordered allocation
    float* d_minMax;
    CUDA_CHECK(cudaMallocAsync(&d_minMax, 2 * sizeof(float), stream));
    
    // Compute statistics (simplified)
    float h_stats[2] = {0.0f, 1.0f};
    CUDA_CHECK(cudaMemcpyAsync(d_minMax, h_stats, 2 * sizeof(float), 
                               cudaMemcpyHostToDevice, stream));
    
    // Launch normalization kernel
    int blockSize = 256;
    int gridSize = (size + blockSize - 1) / blockSize;
    normalizeKernel<<<gridSize, blockSize, 0, stream>>>(d_data, size, h_stats[0], h_stats[1]);
    
    // Free temporary buffer using stream-ordered deallocation
    CUDA_CHECK(cudaFreeAsync(d_minMax, stream));
}

// Stream-ordered utility for type conversion with temporary buffer
void convertAndNormalizeAsync(const unsigned char* d_input, float* d_output, 
                              int size, cudaStream_t stream) {
    // Allocate temporary buffer for intermediate results using stream-ordered allocation
    float* d_temp;
    CUDA_CHECK(cudaMallocAsync(&d_temp, size * sizeof(float), stream));
    
    // Launch conversion kernel
    int blockSize = 256;
    int gridSize = (size + blockSize - 1) / blockSize;
    convertUInt8ToFloatKernel<<<gridSize, blockSize, 0, stream>>>(d_input, d_temp, size);
    
    // Copy to output
    CUDA_CHECK(cudaMemcpyAsync(d_output, d_temp, size * sizeof(float), 
                               cudaMemcpyDeviceToDevice, stream));
    
    // Free temporary buffer using stream-ordered deallocation
    CUDA_CHECK(cudaFreeAsync(d_temp, stream));
}

// Stream-ordered utility for boundary padding with workspace allocation
void applyReflectivePaddingAsync(const float* d_input, float* d_output,
                                 int width, int height, int padSize, 
                                 cudaStream_t stream) {
    // Allocate temporary workspace using stream-ordered allocation
    size_t workspaceSize = (width + 2 * padSize) * (height + 2 * padSize) * sizeof(float);
    float* d_workspace;
    CUDA_CHECK(cudaMallocAsync(&d_workspace, workspaceSize, stream));
    
    // Configure kernel launch
    dim3 blockSize(16, 16);
    dim3 gridSize((width + 2 * padSize + blockSize.x - 1) / blockSize.x,
                  (height + 2 * padSize + blockSize.y - 1) / blockSize.y);
    
    // Launch boundary padding kernel
    reflectBoundaryKernel<<<gridSize, blockSize, 0, stream>>>(
        d_input, d_workspace, width, height, padSize);
    
    // Copy result to output
    CUDA_CHECK(cudaMemcpyAsync(d_output, d_workspace, workspaceSize, 
                               cudaMemcpyDeviceToDevice, stream));
    
    // Free workspace using stream-ordered deallocation
    CUDA_CHECK(cudaFreeAsync(d_workspace, stream));
}

// Template function for generic data scaling with stream-ordered allocation
template<typename T>
void scaleDataAsync(T* d_data, int size, T scaleFactor, cudaStream_t stream) {
    // Allocate temporary buffer using stream-ordered allocation
    T* d_tempBuffer;
    CUDA_CHECK(cudaMallocAsync(&d_tempBuffer, size * sizeof(T), stream));
    
    // Copy data to temporary buffer
    CUDA_CHECK(cudaMemcpyAsync(d_tempBuffer, d_data, size * sizeof(T), 
                               cudaMemcpyDeviceToDevice, stream));
    
    // Process data (simplified - would normally launch a kernel)
    CUDA_CHECK(cudaMemcpyAsync(d_data, d_tempBuffer, size * sizeof(T), 
                               cudaMemcpyDeviceToDevice, stream));
    
    // Free temporary buffer using stream-ordered deallocation
    CUDA_CHECK(cudaFreeAsync(d_tempBuffer, stream));
}

// Explicit template instantiations
template void scaleDataAsync<float>(float*, int, float, cudaStream_t);
template void scaleDataAsync<double>(double*, int, double, cudaStream_t);
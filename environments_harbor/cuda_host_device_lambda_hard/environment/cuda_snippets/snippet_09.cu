// snippet_09.cu
// Valid CUDA C++ showing proper lambda usage within a __device__ function

#include <cuda_runtime.h>

__device__ int device_helper(int x, int y) {
    // Define a __device__ lambda within a __device__ function
    auto multiply = [](int a, int b) __device__ {
        return a * b;
    };
    
    // Use the lambda
    int result = multiply(x, y);
    
    // Chain with another lambda
    auto add_offset = [](int val, int offset) __device__ {
        return val + offset;
    };
    
    return add_offset(result, 10);
}

__device__ float complex_computation(float base) {
    // Nested lambda usage with captures
    float multiplier = 2.5f;
    
    auto scale = [multiplier](float val) __device__ {
        return val * multiplier;
    };
    
    auto square_and_scale = [scale](float v) __device__ {
        float squared = v * v;
        return scale(squared);
    };
    
    return square_and_scale(base);
}

__global__ void test_kernel(int* output, float* foutput) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    
    // Call device functions that use lambdas internally
    output[idx] = device_helper(idx, idx + 1);
    foutput[idx] = complex_computation(static_cast<float>(idx));
}

int main() {
    int* d_output;
    float* d_foutput;
    
    cudaMalloc(&d_output, 256 * sizeof(int));
    cudaMalloc(&d_foutput, 256 * sizeof(float));
    
    test_kernel<<<1, 256>>>(d_output, d_foutput);
    
    cudaFree(d_output);
    cudaFree(d_foutput);
    
    return 0;
}
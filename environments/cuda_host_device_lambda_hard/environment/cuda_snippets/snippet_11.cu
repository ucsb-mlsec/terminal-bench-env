#include <cuda_runtime.h>
#include <stdio.h>

__global__ void generic_lambda_kernel(int* int_data, float* float_data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // Generic lambda with __device__ specifier for type-generic operations
        auto compare_and_select = [=] __device__ (auto a, auto b, auto threshold) {
            return (a > threshold) ? a : b;
        };
        
        // Use with integers
        int int_threshold = 50;
        int_data[idx] = compare_and_select(int_data[idx], int_threshold, 25);
        
        // Use with floats - same lambda works for different types
        float float_threshold = 0.5f;
        float_data[idx] = compare_and_select(float_data[idx], float_threshold, 0.25f);
        
        // Another generic lambda for arithmetic operations
        auto multiply_and_add = [=] __device__ (auto x, auto y, auto z) {
            return x * y + z;
        };
        
        int_data[idx] = multiply_and_add(int_data[idx], 2, 10);
        float_data[idx] = multiply_and_add(float_data[idx], 2.0f, 1.5f);
    }
}

int main() {
    const int n = 256;
    int *d_int_data;
    float *d_float_data;
    
    cudaMalloc(&d_int_data, n * sizeof(int));
    cudaMalloc(&d_float_data, n * sizeof(float));
    
    generic_lambda_kernel<<<(n + 255) / 256, 256>>>(d_int_data, d_float_data, n);
    cudaDeviceSynchronize();
    
    cudaFree(d_int_data);
    cudaFree(d_float_data);
    
    return 0;
}
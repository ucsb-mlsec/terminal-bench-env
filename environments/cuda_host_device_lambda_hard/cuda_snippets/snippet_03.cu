#include <cuda_runtime.h>
#include <stdio.h>

__global__ void kernel_with_lambda(float *data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    auto compute = [] __host__ __device__ (float x) {
        return x * x + 2.0f * x + 1.0f;
    };
    
    if (idx < n) {
        data[idx] = compute(data[idx]);
    }
}

void host_function(float *data, int n) {
    auto compute = [] __host__ __device__ (float x) {
        return x * x + 2.0f * x + 1.0f;
    };
    
    for (int i = 0; i < n; i++) {
        data[i] = compute(data[i]);
    }
}

int main() {
    const int n = 256;
    float *d_data;
    float h_data[n];
    
    for (int i = 0; i < n; i++) {
        h_data[i] = static_cast<float>(i);
    }
    
    cudaMalloc(&d_data, n * sizeof(float));
    cudaMemcpy(d_data, h_data, n * sizeof(float), cudaMemcpyHostToDevice);
    
    kernel_with_lambda<<<4, 64>>>(d_data, n);
    
    cudaMemcpy(h_data, d_data, n * sizeof(float), cudaMemcpyDeviceToHost);
    
    host_function(h_data, 10);
    
    cudaFree(d_data);
    
    return 0;
}
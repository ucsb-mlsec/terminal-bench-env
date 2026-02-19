#include <cuda_runtime.h>
#include <stdio.h>

__global__ void lambdaKernel(int* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    auto multiply = [](int x) __device__ { return x * 2; };
    
    if (idx < n) {
        data[idx] = multiply(data[idx]);
    }
}

int main() {
    const int N = 256;
    int* d_data;
    int h_data[N];
    
    for (int i = 0; i < N; i++) {
        h_data[i] = i;
    }
    
    cudaMalloc(&d_data, N * sizeof(int));
    cudaMemcpy(d_data, h_data, N * sizeof(int), cudaMemcpyHostToDevice);
    
    lambdaKernel<<<(N + 255) / 256, 256>>>(d_data, N);
    
    cudaMemcpy(h_data, d_data, N * sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_data);
    
    return 0;
}
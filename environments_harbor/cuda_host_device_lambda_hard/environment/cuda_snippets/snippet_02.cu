// Invalid CUDA snippet: Lambda missing __device__ specifier when called from device code

#include <cuda_runtime.h>

__global__ void processArray(int* data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < size) {
        // ERROR: This lambda lacks __device__ specifier but is being called from device code
        auto transform = [](int x) {
            return x * 2 + 1;
        };
        
        data[idx] = transform(data[idx]);
    }
}

int main() {
    const int size = 256;
    int* d_data;
    
    cudaMalloc(&d_data, size * sizeof(int));
    
    processArray<<<4, 64>>>(d_data, size);
    
    cudaFree(d_data);
    
    return 0;
}
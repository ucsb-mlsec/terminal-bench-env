#include <cuda_runtime.h>
#include <stdio.h>

#define BLOCK_SIZE 256
#define ARRAY_SIZE 1048576

__global__ void vectorAdd(const float* a, const float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

__global__ void vectorScale(float* data, float scale, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] *= scale;
    }
}

int main() {
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    
    cudaMemPool_t memPool;
    cudaDeviceGetDefaultMemPool(&memPool, 0);
    
    float *d_a, *d_b, *d_c;
    size_t bytes = ARRAY_SIZE * sizeof(float);
    
    // Use stream-ordered memory pool allocation
    cudaMallocFromPoolAsync(&d_a, bytes, memPool, stream);
    cudaMallocFromPoolAsync(&d_b, bytes, memPool, stream);
    cudaMallocFromPoolAsync(&d_c, bytes, memPool, stream);
    
    float *h_a = new float[ARRAY_SIZE];
    float *h_b = new float[ARRAY_SIZE];
    
    for (int i = 0; i < ARRAY_SIZE; i++) {
        h_a[i] = static_cast<float>(i);
        h_b[i] = static_cast<float>(i * 2);
    }
    
    cudaMemcpyAsync(d_a, h_a, bytes, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_b, h_b, bytes, cudaMemcpyHostToDevice, stream);
    
    int numBlocks = (ARRAY_SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE;
    vectorAdd<<<numBlocks, BLOCK_SIZE, 0, stream>>>(d_a, d_b, d_c, ARRAY_SIZE);
    vectorScale<<<numBlocks, BLOCK_SIZE, 0, stream>>>(d_c, 0.5f, ARRAY_SIZE);
    
    cudaStreamSynchronize(stream);
    
    cudaFreeAsync(d_a, stream);
    cudaFreeAsync(d_b, stream);
    cudaFreeAsync(d_c, stream);
    
    cudaStreamDestroy(stream);
    delete[] h_a;
    delete[] h_b;
    
    return 0;
}
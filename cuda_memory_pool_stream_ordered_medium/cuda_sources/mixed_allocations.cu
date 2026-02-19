#include <cuda_runtime.h>
#include <stdio.h>

// Kernel for vector addition
__global__ void vectorAdd(float *a, float *b, float *c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

// Kernel for vector scaling
__global__ void vectorScale(float *a, float scale, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        a[idx] *= scale;
    }
}

// Legacy allocation function using traditional cudaMalloc
void allocateLegacyBuffers(float **d_temp, int size) {
    cudaMalloc((void**)d_temp, size * sizeof(float));
}

// Modern allocation function using pool-based allocation
void allocatePoolBuffers(float **d_output, int size, cudaStream_t stream, cudaMemPool_t memPool) {
    cudaMallocFromPoolAsync((void**)d_output, size * sizeof(float), memPool, stream);
}

int main() {
    const int N = 1024 * 1024;
    const int size = N * sizeof(float);
    
    // Host arrays
    float *h_a, *h_b, *h_c;
    h_a = (float*)malloc(size);
    h_b = (float*)malloc(size);
    h_c = (float*)malloc(size);
    
    // Initialize input arrays
    for (int i = 0; i < N; i++) {
        h_a[i] = i * 1.0f;
        h_b[i] = i * 2.0f;
    }
    
    // Device arrays using traditional allocation (legacy code)
    float *d_a, *d_b;
    cudaMalloc((void**)&d_a, size);
    cudaMalloc((void**)&d_b, size);
    
    // Create stream and memory pool for modern allocation
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    
    cudaMemPool_t memPool;
    cudaDeviceGetDefaultMemPool(&memPool, 0);
    
    // Device array using pool-based allocation (refactored code)
    float *d_c;
    cudaMallocAsync((void**)&d_c, size, stream);
    
    // Copy data to device
    cudaMemcpyAsync(d_a, h_a, size, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_b, h_b, size, cudaMemcpyHostToDevice, stream);
    
    // Launch kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    vectorAdd<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(d_a, d_b, d_c, N);
    
    // Copy result back
    cudaMemcpyAsync(h_c, d_c, size, cudaMemcpyDeviceToHost, stream);
    
    // Synchronize
    cudaStreamSynchronize(stream);
    
    // Cleanup with mixed deallocation patterns
    cudaFree(d_a);  // Traditional free for traditional allocation
    cudaFree(d_b);
    cudaFreeAsync(d_c, stream);  // Async free for async allocation
    
    cudaStreamDestroy(stream);
    free(h_a);
    free(h_b);
    free(h_c);
    
    return 0;
}
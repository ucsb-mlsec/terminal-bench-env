#include <cuda_runtime.h>

// Naive matrix multiplication kernel
// C = A * B
// A is M x K, B is K x N, C is M x N
__global__ void matmul_naive(const float* A, const float* B, float* C, 
                              int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int i = 0; i < K; i++) {
            sum += A[row * K + i] * B[i * N + col];
        }
        C[row * N + col] = sum;
    }
}

// Host function to launch the kernel
void launch_matmul_naive(const float* d_A, const float* d_B, float* d_C,
                         int M, int N, int K) {
    dim3 blockDim(16, 16);
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x,
                 (M + blockDim.y - 1) / blockDim.y);
    
    matmul_naive<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
}
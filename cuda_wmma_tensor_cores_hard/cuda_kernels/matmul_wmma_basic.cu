#include <cuda_fp16.h>
#include <mma.h>
#include <stdio.h>

using namespace nvcuda;

#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

#define MATRIX_M 256
#define MATRIX_N 256
#define MATRIX_K 256

__global__ void matmul_wmma_basic_kernel(half *a, half *b, float *c, int M, int N, int K) {
    int warpM = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
    int warpN = (blockIdx.y * blockDim.y + threadIdx.y);
    
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;
    
    wmma::fill_fragment(acc_frag, 0.0f);
    
    int aRow = warpM * WMMA_M;
    int bCol = warpN * WMMA_N;
    
    if (aRow < M && bCol < N) {
        for (int i = 0; i < K; i += WMMA_K) {
            int aCol = i;
            int bRow = i;
            
            if (aCol < K && bRow < K) {
                wmma::load_matrix_sync(a_frag, a + aRow * K + aCol, K);
                wmma::load_matrix_sync(b_frag, b + bRow + bCol * K, K);
                
                wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
            }
        }
        
        wmma::store_matrix_sync(c + aRow * N + bCol, acc_frag, N, wmma::mem_row_major);
    }
}

void matmul_wmma_basic(half *d_a, half *d_b, float *d_c, int M, int N, int K) {
    dim3 gridDim;
    dim3 blockDim;
    
    blockDim.x = 128;
    blockDim.y = 4;
    
    gridDim.x = (M + (WMMA_M * blockDim.x / warpSize) - 1) / (WMMA_M * blockDim.x / warpSize);
    gridDim.y = (N + (WMMA_N * blockDim.y) - 1) / (WMMA_N * blockDim.y);
    
    matmul_wmma_basic_kernel<<<gridDim, blockDim>>>(d_a, d_b, d_c, M, N, K);
}
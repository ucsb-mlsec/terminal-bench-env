#include <cuda_fp16.h>
#include <mma.h>
#include <cuda_runtime.h>

using namespace nvcuda::wmma;

#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16
#define WARP_SIZE 32

// Advanced matrix multiplication kernel using WMMA with double-buffering optimization
__global__ void matmul_wmma_advanced_kernel(
    const half* __restrict__ A,
    const half* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K)
{
    // Shared memory for double-buffering tiles
    __shared__ half tileA[2][256 * 16];
    __shared__ half tileB[2][16 * 256];
    
    int warpM = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    int warpN = (blockIdx.y * blockDim.y + threadIdx.y);
    
    // Declare WMMA fragments with proper template parameters
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
    
    // Initialize accumulator fragment to zero
    wmma::fill_fragment(acc_frag, 0.0f);
    
    int buffer_idx = 0;
    
    // Main computation loop with double-buffering
    for (int i = 0; i < K; i += WMMA_K) {
        // Load tiles into shared memory (double-buffered)
        int next_buffer = 1 - buffer_idx;
        
        // Preload next tile while computing current
        if (i + WMMA_K < K) {
            for (int t = threadIdx.x; t < 256 * 16; t += blockDim.x) {
                int row = t / 16;
                int col = t % 16;
                if (warpM * 16 + row < M && i + WMMA_K + col < K) {
                    tileA[next_buffer][t] = A[(warpM * 16 + row) * K + i + WMMA_K + col];
                }
            }
            for (int t = threadIdx.x; t < 16 * 256; t += blockDim.x) {
                int row = t / 256;
                int col = t % 256;
                if (i + WMMA_K + row < K && warpN * 16 + col < N) {
                    tileB[next_buffer][t] = B[(i + WMMA_K + row) * N + warpN * 16 + col];
                }
            }
        }
        
        __syncthreads();
        
        // Load current fragments from shared memory
        wmma::load_matrix_sync(a_frag, 
                             &tileA[buffer_idx][0] + (warpM % 16) * 16 * WMMA_K, 
                             WMMA_K);
        
        wmma::load_matrix_sync(b_frag, 
                             &tileB[buffer_idx][0] + (warpN % 16) * 16,
                             16);
        
        // Perform matrix multiply-accumulate operation
        wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
        
        buffer_idx = next_buffer;
        __syncthreads();
    }
    
    // Store the accumulated result back to global memory
    int cRow = warpM * WMMA_M;
    int cCol = warpN * WMMA_N;
    
    if (cRow < M && cCol < N) {
        wmma::store_matrix_sync(C + cRow * N + cCol, acc_frag, N, wmma::mem_row_major);
    }
}

// Host function to launch the advanced WMMA kernel
void matmul_wmma_advanced(
    const half* d_A,
    const half* d_B, 
    float* d_C,
    int M, int N, int K)
{
    // Ensure dimensions are multiples of 16
    assert(M % 16 == 0 && N % 16 == 0 && K % 16 == 0);
    
    dim3 blockDim(128, 4);
    dim3 gridDim((M + WMMA_M - 1) / WMMA_M, (N + WMMA_N - 1) / WMMA_N);
    
    matmul_wmma_advanced_kernel<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
}

// Alternative kernel with pipeline optimization
__global__ void matmul_wmma_pipelined_kernel(
    const half* __restrict__ A,
    const half* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K)
{
    __shared__ half shmemA[256];
    __shared__ half shmemB[256];
    
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag[2];
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag[2];
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc_frag;
    
    wmma::fill_fragment(acc_frag, 0.0f);
    
    int warpRow = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE * 16;
    int warpCol = blockIdx.y * 16;
    
    // Pipeline: load first fragments
    if (warpRow < M && warpCol < N && 0 < K) {
        wmma::load_matrix_sync(a_frag[0], A + warpRow * K, K);
        wmma::load_matrix_sync(b_frag[0], B + warpCol, N);
    }
    
    // Main loop with pipelining
    for (int k = WMMA_K; k < K; k += WMMA_K) {
        int curr = (k / WMMA_K - 1) % 2;
        int next = (k / WMMA_K) % 2;
        
        // Load next while computing current
        if (k < K) {
            wmma::load_matrix_sync(a_frag[next], A + warpRow * K + k, K);
            wmma::load_matrix_sync(b_frag[next], B + k * N + warpCol, N);
        }
        
        wmma::mma_sync(acc_frag, a_frag[curr], b_frag[curr], acc_frag);
    }
    
    // Final computation
    int final_idx = ((K / WMMA_K - 1) % 2);
    wmma::mma_sync(acc_frag, a_frag[final_idx], b_frag[final_idx], acc_frag);
    
    // Store result
    if (warpRow < M && warpCol < N) {
        wmma::store_matrix_sync(C + warpRow * N + warpCol, acc_frag, N, wmma::mem_row_major);
    }
}
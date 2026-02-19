#include <cuda_runtime.h>
#include <mma.h>
#include <cuda_fp16.h>

using namespace nvcuda::wmma;

#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16
#define TILE_SIZE 128

__global__ void matmul_wmma_optimized_kernel(
    const half* __restrict__ A,
    const half* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K)
{
    // Shared memory for tiling
    __shared__ half tileA[TILE_SIZE][TILE_SIZE];
    __shared__ half tileB[TILE_SIZE][TILE_SIZE];
    
    // Warp and thread identification
    int warpM = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
    int warpN = (blockIdx.y * blockDim.y + threadIdx.y);
    
    // WMMA fragment declarations with proper template parameters
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;
    
    // Initialize accumulator fragment to zero
    wmma::fill_fragment(acc_frag, 0.0f);
    
    // Calculate matrix block indices
    int block_row = blockIdx.y * TILE_SIZE;
    int block_col = blockIdx.x * TILE_SIZE;
    
    // Tile accumulation loop
    for (int tile_idx = 0; tile_idx < (K + TILE_SIZE - 1) / TILE_SIZE; ++tile_idx) {
        
        // Load tiles into shared memory
        int tile_k = tile_idx * TILE_SIZE;
        
        for (int i = threadIdx.y; i < TILE_SIZE; i += blockDim.y) {
            for (int j = threadIdx.x; j < TILE_SIZE; j += blockDim.x) {
                int global_row = block_row + i;
                int global_col = tile_k + j;
                
                if (global_row < M && global_col < K) {
                    tileA[i][j] = A[global_row * K + global_col];
                } else {
                    tileA[i][j] = __float2half(0.0f);
                }
                
                global_row = tile_k + i;
                global_col = block_col + j;
                
                if (global_row < K && global_col < N) {
                    tileB[i][j] = B[global_row * N + global_col];
                } else {
                    tileB[i][j] = __float2half(0.0f);
                }
            }
        }
        
        __syncthreads();
        
        // WMMA operations on tile fragments
        for (int k = 0; k < TILE_SIZE; k += WMMA_K) {
            int aRow = (threadIdx.y / 2) * WMMA_M;
            int aCol = k;
            int bRow = k;
            int bCol = (threadIdx.x / 4) * WMMA_N;
            
            // Load matrix fragments from shared memory
            wmma::load_matrix_sync(a_frag, &tileA[aRow][aCol], TILE_SIZE);
            wmma::load_matrix_sync(b_frag, &tileB[bRow][bCol], TILE_SIZE);
            
            // Perform matrix multiply-accumulate
            wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
        }
        
        __syncthreads();
    }
    
    // Store the accumulated result to global memory
    int cRow = block_row + (threadIdx.y / 2) * WMMA_M;
    int cCol = block_col + (threadIdx.x / 4) * WMMA_N;
    
    if (cRow < M && cCol < N) {
        wmma::store_matrix_sync(&C[cRow * N + cCol], acc_frag, N, wmma::mem_row_major);
    }
}

// Host function to launch the optimized WMMA kernel
void launch_matmul_wmma_optimized(
    const half* d_A,
    const half* d_B,
    float* d_C,
    int M, int N, int K)
{
    // Ensure dimensions are multiples of 16
    assert(M % 16 == 0 && N % 16 == 0 && K % 16 == 0);
    
    dim3 blockDim(128, 4);
    dim3 gridDim((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);
    
    matmul_wmma_optimized_kernel<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
}
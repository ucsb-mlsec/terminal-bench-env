#include <cuda_runtime.h>
#include <mma.h>

using namespace nvcuda;

// Broken WMMA kernel - uses incorrect dimensions and missing critical operations
__global__ void matmul_broken_wmma(half* A, half* B, float* C, int M, int N, int K) {
    // STRUCTURAL ISSUE 1: Wrong dimensions - WMMA requires 16x16x16, not 15x15x15
    wmma::fragment<wmma::matrix_a, 15, 15, 15, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 15, 15, 15, half, wmma::col_major> b_frag;
    
    // STRUCTURAL ISSUE 2: Incompatible accumulator type with dimensions
    wmma::fragment<wmma::accumulator, 15, 15, 15, float> c_frag;
    
    int warpM = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
    int warpN = (blockIdx.y * blockDim.y + threadIdx.y);
    
    // STRUCTURAL ISSUE 3: Matrix dimensions don't align with WMMA requirements
    // Using 17 instead of 16 or 32
    const int WMMA_M = 17;
    const int WMMA_N = 17;
    const int WMMA_K = 17;
    
    int aRow = warpM * WMMA_M;
    int bCol = warpN * WMMA_N;
    
    // Initialize accumulator - this part is correct
    wmma::fill_fragment(c_frag, 0.0f);
    
    // Load matrices
    for (int i = 0; i < K; i += WMMA_K) {
        int aCol = i;
        int bRow = i;
        
        // STRUCTURAL ISSUE 4: Load operations present but...
        wmma::load_matrix_sync(a_frag, A + aRow * K + aCol, K);
        wmma::load_matrix_sync(b_frag, B + bRow * N + bCol, N);
        
        // STRUCTURAL ISSUE 5: MISSING mma_sync - the critical multiply-accumulate operation!
        // This is what actually performs the Tensor Core computation
        // wmma::mma_sync(c_frag, a_frag, b_frag, c_frag); // MISSING!
    }
    
    // STRUCTURAL ISSUE 6: Store operation present but mma_sync was missing
    // So c_frag still contains zeros
    wmma::store_matrix_sync(C + aRow * N + bCol, c_frag, N, wmma::mem_row_major);
}

// Another broken kernel with type mismatches
__global__ void matmul_type_mismatch(double* A, float* B, half* C, int M, int N, int K) {
    // STRUCTURAL ISSUE 7: Declaring fragments for half but input is double/float
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, half> c_frag;
    
    int warpM = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
    int warpN = (blockIdx.y * blockDim.y + threadIdx.y);
    
    // STRUCTURAL ISSUE 8: Type incompatibility - can't load double into half fragment
    wmma::fill_fragment(c_frag, __float2half(0.0f));
    
    // This would fail - type mismatch
    wmma::load_matrix_sync(a_frag, (half*)A, K); // casting doesn't fix structural issue
    wmma::load_matrix_sync(b_frag, (half*)B, N);
    
    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    wmma::store_matrix_sync(C, c_frag, N, wmma::mem_row_major);
}

// Kernel with unused fragments - structural issue
__global__ void matmul_unused_fragments(half* A, half* B, float* C, int M, int N, int K) {
    // STRUCTURAL ISSUE 9: Fragments declared but never used properly
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // STRUCTURAL ISSUE 10: Falling back to scalar operations instead of using WMMA
    if (idx < M * N) {
        int row = idx / N;
        int col = idx % N;
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += __half2float(A[row * K + k]) * __half2float(B[k * N + col]);
        }
        C[row * N + col] = sum;
    }
    
    // Fragments declared above but never loaded, computed, or stored!
}
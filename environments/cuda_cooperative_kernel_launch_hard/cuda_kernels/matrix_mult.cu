#include <cuda_runtime.h>
#include <stdio.h>

// Matrix multiplication kernel - standard implementation without cooperative groups
// Computes C = A * B where all matrices are square of size N x N
// Uses shared memory tiling for improved performance

#define TILE_SIZE 16

__global__ void matrixMultKernel(float *A, float *B, float *C, int N) {
    // Shared memory for tiling
    __shared__ float tileA[TILE_SIZE][TILE_SIZE];
    __shared__ float tileB[TILE_SIZE][TILE_SIZE];
    
    // Calculate global row and column indices
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    
    float sum = 0.0f;
    
    // Loop over tiles
    int numTiles = (N + TILE_SIZE - 1) / TILE_SIZE;
    
    for (int t = 0; t < numTiles; t++) {
        // Load elements into shared memory
        int aCol = t * TILE_SIZE + threadIdx.x;
        int bRow = t * TILE_SIZE + threadIdx.y;
        
        if (row < N && aCol < N) {
            tileA[threadIdx.y][threadIdx.x] = A[row * N + aCol];
        } else {
            tileA[threadIdx.y][threadIdx.x] = 0.0f;
        }
        
        if (bRow < N && col < N) {
            tileB[threadIdx.y][threadIdx.x] = B[bRow * N + col];
        } else {
            tileB[threadIdx.y][threadIdx.x] = 0.0f;
        }
        
        // Synchronize to ensure all data is loaded
        __syncthreads();
        
        // Compute partial product for this tile
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];
        }
        
        // Synchronize before loading next tile
        __syncthreads();
    }
    
    // Write result to global memory
    if (row < N && col < N) {
        C[row * N + col] = sum;
    }
}

// Host function to launch the matrix multiplication kernel
void launchMatrixMult(float *d_A, float *d_B, float *d_C, int N) {
    // Configure kernel launch parameters
    dim3 blockDim(TILE_SIZE, TILE_SIZE);
    dim3 gridDim((N + TILE_SIZE - 1) / TILE_SIZE, (N + TILE_SIZE - 1) / TILE_SIZE);
    
    // Launch kernel using standard syntax
    matrixMultKernel<<<gridDim, blockDim>>>(d_A, d_B, d_C, N);
    
    // Check for launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Kernel launch failed: %s\n", cudaGetErrorString(err));
    }
}
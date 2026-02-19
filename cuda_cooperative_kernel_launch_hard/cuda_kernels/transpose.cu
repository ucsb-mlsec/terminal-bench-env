#include <cuda_runtime.h>
#include <stdio.h>

#define TILE_DIM 32
#define BLOCK_ROWS 8

__global__ void transposeKernel(float* input, float* output, int rows, int cols) {
    __shared__ float tile[TILE_DIM][TILE_DIM + 1];
    
    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;
    
    // Load data into shared memory tile
    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        int row = y + j;
        if (x < cols && row < rows) {
            tile[threadIdx.y + j][threadIdx.x] = input[row * cols + x];
        }
    }
    
    __syncthreads();
    
    // Transpose block coordinates
    x = blockIdx.y * TILE_DIM + threadIdx.x;
    y = blockIdx.x * TILE_DIM + threadIdx.y;
    
    // Write transposed data to output
    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        int row = y + j;
        if (x < rows && row < cols) {
            output[row * rows + x] = tile[threadIdx.x][threadIdx.y + j];
        }
    }
}

void launchTranspose(float* d_input, float* d_output, int rows, int cols) {
    dim3 blockDim(TILE_DIM, BLOCK_ROWS);
    dim3 gridDim((cols + TILE_DIM - 1) / TILE_DIM, 
                 (rows + TILE_DIM - 1) / TILE_DIM);
    
    transposeKernel<<<gridDim, blockDim>>>(d_input, d_output, rows, cols);
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Kernel launch failed: %s\n", cudaGetErrorString(err));
    }
}

int main() {
    const int rows = 1024;
    const int cols = 2048;
    const int size = rows * cols * sizeof(float);
    
    float *h_input = (float*)malloc(size);
    float *h_output = (float*)malloc(size);
    float *d_input, *d_output;
    
    // Initialize input matrix
    for (int i = 0; i < rows * cols; i++) {
        h_input[i] = (float)i;
    }
    
    cudaMalloc(&d_input, size);
    cudaMalloc(&d_output, size);
    
    cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);
    
    launchTranspose(d_input, d_output, rows, cols);
    
    cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost);
    
    cudaFree(d_input);
    cudaFree(d_output);
    free(h_input);
    free(h_output);
    
    return 0;
}
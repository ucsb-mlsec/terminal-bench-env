#include <cuda_runtime.h>
#include <nvToolsExt.h>
#include <stdio.h>

#define TILE_WIDTH 16

__global__ void matrixMultiplyKernel(float *A, float *B, float *C, int width) {
    __shared__ float ds_A[TILE_WIDTH][TILE_WIDTH];
    __shared__ float ds_B[TILE_WIDTH][TILE_WIDTH];
    
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    int row = by * TILE_WIDTH + ty;
    int col = bx * TILE_WIDTH + tx;
    
    float sum = 0.0f;
    
    for (int m = 0; m < (width + TILE_WIDTH - 1) / TILE_WIDTH; m++) {
        if (row < width && (m * TILE_WIDTH + tx) < width) {
            ds_A[ty][tx] = A[row * width + m * TILE_WIDTH + tx];
        } else {
            ds_A[ty][tx] = 0.0f;
        }
        
        if ((m * TILE_WIDTH + ty) < width && col < width) {
            ds_B[ty][tx] = B[(m * TILE_WIDTH + ty) * width + col];
        } else {
            ds_B[ty][tx] = 0.0f;
        }
        
        __syncthreads();
        
        for (int k = 0; k < TILE_WIDTH; k++) {
            sum += ds_A[ty][k] * ds_B[k][tx];
        }
        
        __syncthreads();
    }
    
    if (row < width && col < width) {
        C[row * width + col] = sum;
    }
}

void matrixMultiply(float *h_A, float *h_B, float *h_C, int width) {
    float *d_A, *d_B, *d_C;
    size_t size = width * width * sizeof(float);
    
    nvtxRangePush("GPU Memory Allocation");
    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_B, size);
    cudaMalloc((void**)&d_C, size);
    nvtxRangePop();
    
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);
    
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
    dim3 dimGrid((width + TILE_WIDTH - 1) / TILE_WIDTH, 
                 (width + TILE_WIDTH - 1) / TILE_WIDTH);
    
    nvtxRangePush("Matrix Multiply Kernel");
    matrixMultiplyKernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, width);
    cudaDeviceSynchronize();
    nvtxRangePop();
    
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
    
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

int main() {
    int width = 1024;
    size_t size = width * width * sizeof(float);
    
    float *h_A = (float*)malloc(size);
    float *h_B = (float*)malloc(size);
    float *h_C = (float*)malloc(size);
    
    for (int i = 0; i < width * width; i++) {
        h_A[i] = 1.0f;
        h_B[i] = 2.0f;
    }
    
    matrixMultiply(h_A, h_B, h_C, width);
    
    printf("Matrix multiplication completed. Result[0][0] = %f\n", h_C[0]);
    
    free(h_A);
    free(h_B);
    free(h_C);
    
    return 0;
}
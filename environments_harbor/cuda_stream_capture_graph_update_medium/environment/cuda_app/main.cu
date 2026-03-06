#include <cuda_runtime.h>
#include <stdio.h>

#define N 1024
#define ITERATIONS 100

// Simple vector addition kernel
__global__ void vectorAdd(float *a, float *b, float *c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

// CUDA error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

int main() {
    float *h_a, *h_b, *h_c;
    float *d_a, *d_b, *d_c;
    size_t bytes = N * sizeof(float);
    
    // Allocate host memory
    h_a = (float*)malloc(bytes);
    h_b = (float*)malloc(bytes);
    h_c = (float*)malloc(bytes);
    
    // Initialize host arrays
    for (int i = 0; i < N; i++) {
        h_a[i] = (float)i;
        h_b[i] = (float)(i * 2);
    }
    
    // Allocate device memory
    CUDA_CHECK(cudaMalloc(&d_a, bytes));
    CUDA_CHECK(cudaMalloc(&d_b, bytes));
    CUDA_CHECK(cudaMalloc(&d_c, bytes));
    
    // Copy data to device
    CUDA_CHECK(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice));
    
    // Create a stream for graph capture
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    
    // Kernel launch parameters
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    
    printf("Starting computational pipeline with %d iterations\n", ITERATIONS);
    
    // INEFFICIENT IMPLEMENTATION: Graph created and destroyed in each iteration
    for (int iter = 0; iter < ITERATIONS; iter++) {
        cudaGraph_t graph;
        cudaGraphExec_t graphExec;
        
        // Create graph (INSIDE LOOP - INEFFICIENT!)
        CUDA_CHECK(cudaGraphCreate(&graph, 0));
        
        // Begin capture
        CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
        
        // Launch kernel during capture
        vectorAdd<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(d_a, d_b, d_c, N);
        
        // End capture
        CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
        
        // Instantiate the graph
        CUDA_CHECK(cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0));
        
        // Launch the graph
        CUDA_CHECK(cudaGraphLaunch(graphExec, stream));
        
        // Synchronize stream
        CUDA_CHECK(cudaStreamSynchronize(stream));
        
        // Destroy graph exec and graph (INSIDE LOOP - INEFFICIENT!)
        CUDA_CHECK(cudaGraphExecDestroy(graphExec));
        CUDA_CHECK(cudaGraphDestroy(graph));
        
        if ((iter + 1) % 10 == 0) {
            printf("Completed iteration %d/%d\n", iter + 1, ITERATIONS);
        }
    }
    
    printf("Pipeline execution complete\n");
    
    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost));
    
    // Verify results (spot check)
    bool success = true;
    for (int i = 0; i < 10; i++) {
        float expected = h_a[i] + h_b[i];
        if (h_c[i] != expected) {
            printf("Verification failed at index %d: expected %f, got %f\n", 
                   i, expected, h_c[i]);
            success = false;
            break;
        }
    }
    
    if (success) {
        printf("Verification passed!\n");
    }
    
    // Cleanup
    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));
    
    free(h_a);
    free(h_b);
    free(h_c);
    
    printf("Resources cleaned up successfully\n");
    
    return 0;
}
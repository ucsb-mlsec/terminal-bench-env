#include <stdio.h>
#include <cuda_runtime.h>

#define DATA_SIZE 1048576
#define NUM_STREAMS 3
#define ELEMENTS_PER_STREAM (DATA_SIZE / NUM_STREAMS)

// Simple CUDA kernel for data scaling
__global__ void scaleKernel(float *data, float scale_factor, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = data[idx] * scale_factor;
    }
}

// CUDA kernel for data transformation
__global__ void transformKernel(float *input, float *output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = input[idx] * 2.0f + 1.0f;
    }
}

int main() {
    printf("Stream-based concurrent processing with pinned memory\n");
    printf("Processing %d elements across %d streams\n", DATA_SIZE, NUM_STREAMS);

    // Create CUDA streams for concurrent execution
    cudaStream_t streams[NUM_STREAMS];
    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaStreamCreate(&streams[i]);
    }

    // Pinned memory is required for asynchronous operations
    // Using cudaHostAlloc() allocates page-locked (pinned) memory
    // This enables Direct Memory Access (DMA) and async transfers
    float *h_input_data;
    float *h_output_data;
    float *h_intermediate_data;

    size_t total_bytes = DATA_SIZE * sizeof(float);
    size_t stream_bytes = ELEMENTS_PER_STREAM * sizeof(float);

    // Allocate pinned host memory for input data
    // cudaHostAllocDefault creates page-locked memory for optimal bandwidth
    cudaHostAlloc(&h_input_data, total_bytes, cudaHostAllocDefault);
    if (h_input_data == NULL) {
        fprintf(stderr, "Failed to allocate pinned input memory\n");
        return -1;
    }

    // Allocate pinned host memory for output data
    // Pinned memory eliminates extra copy operations during transfers
    cudaHostAlloc(&h_output_data, total_bytes, cudaHostAllocDefault);
    if (h_output_data == NULL) {
        fprintf(stderr, "Failed to allocate pinned output memory\n");
        cudaFreeHost(h_input_data);
        return -1;
    }

    // Allocate pinned host memory for intermediate results
    // Required for async transfers in stream processing pipeline
    cudaHostAlloc(&h_intermediate_data, total_bytes, cudaHostAllocDefault);
    if (h_intermediate_data == NULL) {
        fprintf(stderr, "Failed to allocate pinned intermediate memory\n");
        cudaFreeHost(h_input_data);
        cudaFreeHost(h_output_data);
        return -1;
    }

    // Initialize input data
    printf("Initializing input data...\n");
    for (int i = 0; i < DATA_SIZE; i++) {
        h_input_data[i] = (float)i * 0.5f;
    }

    // Allocate device memory buffers for each stream
    float *d_input[NUM_STREAMS];
    float *d_output[NUM_STREAMS];

    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaMalloc(&d_input[i], stream_bytes);
        cudaMalloc(&d_output[i], stream_bytes);
    }

    // Configure kernel launch parameters
    int threads_per_block = 256;
    int blocks_per_stream = (ELEMENTS_PER_STREAM + threads_per_block - 1) / threads_per_block;

    printf("Starting stream-based processing...\n");

    // Launch async operations on each stream
    // This enables concurrent execution and optimal GPU utilization
    for (int i = 0; i < NUM_STREAMS; i++) {
        int offset = i * ELEMENTS_PER_STREAM;
        
        // Async copy from host to device (requires pinned memory)
        cudaMemcpyAsync(d_input[i], 
                       h_input_data + offset, 
                       stream_bytes, 
                       cudaMemcpyHostToDevice, 
                       streams[i]);

        // Launch scaling kernel on this stream
        scaleKernel<<<blocks_per_stream, threads_per_block, 0, streams[i]>>>(
            d_input[i], 2.5f, ELEMENTS_PER_STREAM);

        // Launch transformation kernel
        transformKernel<<<blocks_per_stream, threads_per_block, 0, streams[i]>>>(
            d_input[i], d_output[i], ELEMENTS_PER_STREAM);

        // Async copy from device to host (requires pinned memory)
        cudaMemcpyAsync(h_output_data + offset, 
                       d_output[i], 
                       stream_bytes, 
                       cudaMemcpyDeviceToHost, 
                       streams[i]);
    }

    // Synchronize all streams to ensure completion
    printf("Synchronizing streams...\n");
    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaStreamSynchronize(streams[i]);
    }

    printf("Stream processing complete!\n");

    // Second processing pass using intermediate buffer
    printf("Starting second processing pass...\n");
    for (int i = 0; i < NUM_STREAMS; i++) {
        int offset = i * ELEMENTS_PER_STREAM;
        
        // Copy output to intermediate buffer for additional processing
        cudaMemcpyAsync(d_input[i], 
                       h_output_data + offset, 
                       stream_bytes, 
                       cudaMemcpyHostToDevice, 
                       streams[i]);

        // Apply additional scaling
        scaleKernel<<<blocks_per_stream, threads_per_block, 0, streams[i]>>>(
            d_input[i], 0.5f, ELEMENTS_PER_STREAM);

        // Copy results to intermediate host buffer
        cudaMemcpyAsync(h_intermediate_data + offset, 
                       d_input[i], 
                       stream_bytes, 
                       cudaMemcpyDeviceToHost, 
                       streams[i]);
    }

    // Wait for all operations to complete
    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaStreamSynchronize(streams[i]);
    }

    printf("Second pass complete!\n");

    // Verify a few results
    printf("Sample results (first 5 elements):\n");
    for (int i = 0; i < 5; i++) {
        printf("  Input[%d]: %.2f -> Output[%d]: %.2f -> Intermediate[%d]: %.2f\n",
               i, h_input_data[i], i, h_output_data[i], i, h_intermediate_data[i]);
    }

    // Cleanup device memory
    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaFree(d_input[i]);
        cudaFree(d_output[i]);
    }

    // Cleanup pinned host memory
    // Use cudaFreeHost() to free memory allocated with cudaHostAlloc()
    cudaFreeHost(h_input_data);
    cudaFreeHost(h_output_data);
    cudaFreeHost(h_intermediate_data);

    // Destroy streams
    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaStreamDestroy(streams[i]);
    }

    printf("All resources cleaned up successfully\n");
    return 0;
}
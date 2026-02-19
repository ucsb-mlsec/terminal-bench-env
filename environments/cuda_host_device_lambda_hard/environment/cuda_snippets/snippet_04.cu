// Invalid CUDA code: Device lambda capturing host variable by reference
// This will fail to compile because device code cannot access host memory

#include <cuda_runtime.h>
#include <stdio.h>

__global__ void kernel_with_invalid_lambda(int* d_output, int size) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (tid < size) {
        // This lambda is executed in device code but tries to capture
        // a device-local variable by reference, which is then used incorrectly
        int host_value = 42;
        
        auto device_lambda = [&]() __device__ {
            // Attempting to access captured reference in device code
            return host_value * 2;
        };
        
        d_output[tid] = device_lambda();
    }
}

void test_invalid_capture() {
    // Host variable that will be incorrectly captured
    int host_counter = 100;
    
    int* d_data;
    cudaMalloc(&d_data, 256 * sizeof(int));
    
    // This lambda tries to capture host variable by reference for device use
    auto invalid_lambda = [&host_counter]() __device__ {
        // ERROR: Cannot capture host variable by reference for device execution
        return host_counter + threadIdx.x;
    };
    
    // Attempting to use the lambda in kernel context
    kernel_with_invalid_lambda<<<1, 256>>>(d_data, 256);
    
    cudaFree(d_data);
}

int main() {
    test_invalid_capture();
    return 0;
}
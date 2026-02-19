// snippet_10.cu - Invalid: Device lambda attempting to call host-only function

#include <cuda_runtime.h>
#include <iostream>

// Host-only function (no __device__ attribute)
void hostOnlyFunction(int value) {
    std::cout << "Host value: " << value << std::endl;
}

__global__ void kernelWithInvalidLambda(int* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        // Invalid: This lambda is marked __device__ but tries to call
        // a host-only function (hostOnlyFunction)
        auto deviceLambda = [=] __device__ (int val) {
            // ERROR: Calling host-only function from device code
            hostOnlyFunction(val);
            return val * 2;
        };
        
        data[idx] = deviceLambda(data[idx]);
    }
}

int main() {
    const int n = 256;
    int* d_data;
    
    cudaMalloc(&d_data, n * sizeof(int));
    
    kernelWithInvalidLambda<<<4, 64>>>(d_data, n);
    
    cudaFree(d_data);
    return 0;
}

// Expected compilation error:
// error: calling a __host__ function from a __host__ __device__ function is not allowed
// or similar error indicating that hostOnlyFunction cannot be called from device code
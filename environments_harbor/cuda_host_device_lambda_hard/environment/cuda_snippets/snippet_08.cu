// Invalid: Using std::function with device lambda causes compilation error
// std::function is a host-only construct and cannot be used in device code

#include <cuda_runtime.h>
#include <functional>

template<typename Func>
__global__ void kernel_with_function(Func f, int* data) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    data[idx] = f(data[idx]);
}

__global__ void device_kernel(int* data) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    
    // ERROR: std::function cannot be instantiated in device code
    // std::function requires dynamic memory allocation and RTTI
    // which are not available in device code context
    std::function<int(int)> transform = [=] __device__ (int x) {
        return x * 2;
    };
    
    data[idx] = transform(data[idx]);
}

void launch_kernel() {
    int* d_data;
    cudaMalloc(&d_data, 256 * sizeof(int));
    
    // This will fail to compile because std::function
    // cannot be used in device code
    device_kernel<<<1, 256>>>(d_data);
    
    cudaFree(d_data);
}
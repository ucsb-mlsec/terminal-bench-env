// Invalid CUDA snippet: Lambda used in device code without __device__ attribute
// This will fail to compile because the lambda is passed to device code
// but lacks the necessary __device__ annotation

#include <cuda_runtime.h>

template<typename Func>
__device__ void apply_functor(Func f, int value) {
    f(value);
}

__global__ void kernel_with_functor(int* output) {
    int idx = threadIdx.x;
    
    // ERROR: This lambda lacks __device__ attribute but is used in device code
    auto process = [](int x) {
        return x * 2;
    };
    
    // This will fail compilation - passing host-only lambda to device function
    apply_functor(process, idx);
    
    output[idx] = idx;
}

int main() {
    int* d_output;
    cudaMalloc(&d_output, 256 * sizeof(int));
    
    kernel_with_functor<<<1, 256>>>(d_output);
    
    cudaFree(d_output);
    return 0;
}
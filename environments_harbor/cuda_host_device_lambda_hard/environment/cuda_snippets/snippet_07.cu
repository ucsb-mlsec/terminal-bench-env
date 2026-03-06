#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/execution_policy.h>

__global__ void kernel_using_lambda() {
    auto lambda = [] __device__ (int x) {
        return x * x;
    };
    
    int result = lambda(5);
}

int main() {
    const int N = 1000;
    thrust::device_vector<int> input(N);
    thrust::device_vector<int> output(N);
    
    // Initialize input
    thrust::sequence(input.begin(), input.end(), 1);
    
    // Use lambda with thrust::transform
    thrust::transform(
        thrust::device,
        input.begin(),
        input.end(),
        output.begin(),
        [] __host__ __device__ (int x) {
            return x * x + 2 * x + 1;
        }
    );
    
    // Another example with thrust::for_each
    thrust::for_each(
        thrust::device,
        output.begin(),
        output.end(),
        [] __device__ (int& x) {
            x = x / 2;
        }
    );
    
    return 0;
}
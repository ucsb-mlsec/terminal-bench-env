__device__ void process_data(int* output, int size) {
    int multiplier = 10;
    int offset = 5;
    
    auto compute = [=] __device__ (int idx) {
        return (idx * multiplier) + offset;
    };
    
    for (int i = 0; i < size; i++) {
        output[i] = compute(i);
    }
}

__global__ void kernel_with_device_lambda(int* data, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < n) {
        int base_value = 100;
        int scale = 3;
        
        auto transform = [base_value, scale] __device__ (int x) {
            return base_value + (x * scale);
        };
        
        data[tid] = transform(data[tid]);
    }
}

void host_function() {
    int* d_data;
    int n = 1024;
    
    cudaMalloc(&d_data, n * sizeof(int));
    
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    
    kernel_with_device_lambda<<<blocks, threads>>>(d_data, n);
    
    cudaFree(d_data);
}
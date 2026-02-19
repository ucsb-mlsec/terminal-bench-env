#include <iostream>
#include <cuda_runtime.h>

// External CUDA kernel function
extern void launchKernel(float* data, int size);

int main() {
    std::cout << "ML Project starting..." << std::endl;
    
    // Check for available GPUs
    int deviceCount = 0;
    cudaError_t error = cudaGetDeviceCount(&deviceCount);
    
    if (error != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(error) << std::endl;
        return 1;
    }
    
    std::cout << "Number of GPUs found: " << deviceCount << std::endl;
    
    if (deviceCount == 0) {
        std::cerr << "No CUDA-capable devices found!" << std::endl;
        return 1;
    }
    
    // Allocate host memory
    const int dataSize = 1024;
    float* hostData = new float[dataSize];
    
    // Initialize data
    for (int i = 0; i < dataSize; i++) {
        hostData[i] = static_cast<float>(i);
    }
    
    std::cout << "Launching CUDA kernel..." << std::endl;
    
    // Call the CUDA kernel
    launchKernel(hostData, dataSize);
    
    // Check for kernel launch errors
    error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::cerr << "Kernel launch error: " << cudaGetErrorString(error) << std::endl;
        delete[] hostData;
        return 1;
    }
    
    // Wait for GPU to finish
    cudaDeviceSynchronize();
    
    std::cout << "ML Project completed successfully!" << std::endl;
    
    // Cleanup
    delete[] hostData;
    
    return 0;
}
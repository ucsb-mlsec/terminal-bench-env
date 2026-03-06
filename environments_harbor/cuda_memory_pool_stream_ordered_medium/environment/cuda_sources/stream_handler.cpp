#include <cuda_runtime.h>
#include <iostream>
#include <vector>

class StreamHandler {
private:
    cudaStream_t stream;
    cudaMemPool_t memPool;
    
public:
    StreamHandler() {
        cudaStreamCreate(&stream);
        int device;
        cudaGetDevice(&device);
        cudaDeviceGetDefaultMemPool(&memPool, device);
    }
    
    ~StreamHandler() {
        cudaStreamDestroy(stream);
    }
    
    cudaStream_t getStream() const {
        return stream;
    }
    
    void* allocateAsync(size_t size) {
        void* ptr = nullptr;
        cudaError_t err = cudaMallocAsync(&ptr, size, stream);
        if (err != cudaSuccess) {
            std::cerr << "cudaMallocAsync failed: " << cudaGetErrorString(err) << std::endl;
            return nullptr;
        }
        return ptr;
    }
    
    void freeAsync(void* ptr) {
        if (ptr != nullptr) {
            cudaFreeAsync(ptr, stream);
        }
    }
    
    void synchronize() {
        cudaStreamSynchronize(stream);
    }
};

void processData(StreamHandler& handler, size_t dataSize) {
    float* d_data = static_cast<float*>(handler.allocateAsync(dataSize * sizeof(float)));
    
    if (d_data == nullptr) {
        std::cerr << "Failed to allocate device memory" << std::endl;
        return;
    }
    
    // Perform operations on the stream
    // ... kernel launches would go here ...
    
    handler.synchronize();
    handler.freeAsync(d_data);
}

int main() {
    StreamHandler handler;
    processData(handler, 1024 * 1024);
    return 0;
}
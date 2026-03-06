#include <cuda_runtime.h>
#include <iostream>
#include <vector>

class MemoryPoolManager {
private:
    cudaMemPool_t memPool;
    cudaStream_t stream;
    std::vector<void*> allocatedBuffers;

public:
    MemoryPoolManager() {
        cudaStreamCreate(&stream);
        
        // Get default memory pool for device 0
        int device = 0;
        cudaDeviceGetDefaultMemPool(&memPool, device);
        
        // Set memory pool threshold
        uint64_t threshold = UINT64_MAX;
        cudaMemPoolSetAttribute(memPool, cudaMemPoolAttrReleaseThreshold, &threshold);
    }

    ~MemoryPoolManager() {
        cleanup();
        cudaStreamDestroy(stream);
    }

    void* allocateAsync(size_t size) {
        void* devPtr = nullptr;
        cudaError_t err = cudaMallocAsync(&devPtr, size, stream);
        
        if (err != cudaSuccess) {
            std::cerr << "cudaMallocAsync failed: " << cudaGetErrorString(err) << std::endl;
            return nullptr;
        }
        
        allocatedBuffers.push_back(devPtr);
        return devPtr;
    }

    void freeAsync(void* devPtr) {
        if (devPtr != nullptr) {
            cudaFreeAsync(devPtr, stream);
        }
    }

    void synchronize() {
        cudaStreamSynchronize(stream);
    }

    void cleanup() {
        for (void* ptr : allocatedBuffers) {
            cudaFreeAsync(ptr, stream);
        }
        allocatedBuffers.clear();
        synchronize();
    }

    cudaStream_t getStream() const {
        return stream;
    }
};

void processDataWithPool(float* hostData, size_t numElements) {
    MemoryPoolManager poolMgr;
    
    size_t bytes = numElements * sizeof(float);
    float* deviceData = static_cast<float*>(poolMgr.allocateAsync(bytes));
    
    if (deviceData) {
        cudaMemcpyAsync(deviceData, hostData, bytes, cudaMemcpyHostToDevice, poolMgr.getStream());
        poolMgr.synchronize();
        poolMgr.freeAsync(deviceData);
    }
}
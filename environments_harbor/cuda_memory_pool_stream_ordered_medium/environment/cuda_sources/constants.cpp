// constants.cpp
// Configuration constants and utility values for CUDA applications

#include <cstddef>

namespace CudaConfig {
    // Thread block dimensions
    const int BLOCK_SIZE_X = 256;
    const int BLOCK_SIZE_Y = 16;
    const int BLOCK_SIZE_Z = 1;
    
    // Grid dimensions
    const int MAX_GRID_SIZE = 65535;
    const int MIN_GRID_SIZE = 1;
    
    // Memory alignment
    const size_t MEMORY_ALIGNMENT = 256;
    const size_t CACHE_LINE_SIZE = 128;
    
    // Performance tuning parameters
    const int WARP_SIZE = 32;
    const int MAX_THREADS_PER_BLOCK = 1024;
    const int MAX_SHARED_MEMORY_PER_BLOCK = 49152;
    
    // Application-specific constants
    const float EPSILON = 1e-6f;
    const double PI = 3.14159265358979323846;
    const int DEFAULT_STREAM_PRIORITY = 0;
}

// Helper function to compute aligned size
size_t getAlignedSize(size_t size, size_t alignment) {
    return ((size + alignment - 1) / alignment) * alignment;
}

// Helper function to calculate grid dimensions
int calculateGridSize(int dataSize, int blockSize) {
    return (dataSize + blockSize - 1) / blockSize;
}

// Version information
const char* CUDA_APP_VERSION = "2.1.0";
const int VERSION_MAJOR = 2;
const int VERSION_MINOR = 1;
const int VERSION_PATCH = 0;
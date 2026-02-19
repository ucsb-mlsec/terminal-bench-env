#include <cuda/barrier>
#include <cuda/std/barrier>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

// Kernel 1: Simple vector addition with async copy - CORRECT implementation
__global__ void vectorAddCorrect(const float* __restrict__ input, float* __restrict__ output, int n) {
    __shared__ float shared_data[256];
    __shared__ cuda::barrier<cuda::thread_scope_block> barrier;
    
    auto block = cg::this_thread_block();
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (threadIdx.x == 0) {
        init(&barrier, blockDim.x);
    }
    block.sync();
    
    if (tid < n) {
        cuda::memcpy_async(block, &shared_data[threadIdx.x], &input[tid], sizeof(float), barrier);
    }
    
    barrier.arrive_and_wait();
    
    if (tid < n) {
        output[tid] = shared_data[threadIdx.x] + 1.0f;
    }
}

// Kernel 2: Matrix tile processing - MISSING barrier wait
__global__ void matrixTileProcess(const float* __restrict__ input, float* __restrict__ output, int width) {
    __shared__ float tile[16][16];
    __shared__ cuda::barrier<cuda::thread_scope_block> barrier;
    
    auto block = cg::this_thread_block();
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * 16 + ty;
    int col = blockIdx.x * 16 + tx;
    
    if (tx == 0 && ty == 0) {
        init(&barrier, blockDim.x * blockDim.y);
    }
    block.sync();
    
    if (row < width && col < width) {
        cuda::memcpy_async(block, &tile[ty][tx], &input[row * width + col], sizeof(float), barrier);
    }
    
    // BUG: Missing barrier.arrive_and_wait() here
    
    if (row < width && col < width) {
        output[row * width + col] = tile[ty][tx] * 2.0f;
    }
}

// Kernel 3: Stencil operation with halo - INCORRECT thread count
__global__ void stencilCompute(const float* __restrict__ input, float* __restrict__ output, int n) {
    __shared__ float shared_data[258];
    __shared__ cuda::barrier<cuda::thread_scope_block> barrier;
    
    auto block = cg::this_thread_block();
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (threadIdx.x == 0) {
        // BUG: Incorrect thread count - should be blockDim.x, not blockDim.x / 2
        init(&barrier, blockDim.x / 2);
    }
    block.sync();
    
    if (tid < n) {
        cuda::memcpy_async(block, &shared_data[threadIdx.x + 1], &input[tid], sizeof(float), barrier);
    }
    
    barrier.arrive_and_wait();
    
    if (tid < n && tid > 0 && tid < n - 1) {
        output[tid] = (shared_data[threadIdx.x] + shared_data[threadIdx.x + 1] + shared_data[threadIdx.x + 2]) / 3.0f;
    }
}

// Kernel 4: Data transformation - MISSING barrier arrive
__global__ void dataTransform(const int* __restrict__ input, int* __restrict__ output, int n) {
    __shared__ int shared_buffer[512];
    __shared__ cuda::barrier<cuda::thread_scope_block> barrier;
    
    auto block = cg::this_thread_block();
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (threadIdx.x == 0) {
        init(&barrier, blockDim.x);
    }
    block.sync();
    
    if (tid < n) {
        cuda::memcpy_async(block, &shared_buffer[threadIdx.x], &input[tid], sizeof(int), barrier);
    }
    
    // BUG: Missing barrier.arrive_and_wait() - jumps directly to computation
    
    if (tid < n) {
        output[tid] = shared_buffer[threadIdx.x] << 2;
    }
}

// Kernel 5: Reduction with async copy - BARRIER scope mismatch
__global__ void reductionKernel(const float* __restrict__ input, float* __restrict__ output, int n) {
    __shared__ float shared_data[128];
    auto block = cg::this_thread_block();
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (tid < n) {
        if (threadIdx.x < 64) {
            // BUG: Barrier declared inside conditional but used by threads outside
            __shared__ cuda::barrier<cuda::thread_scope_block> barrier;
            if (threadIdx.x == 0) {
                init(&barrier, blockDim.x);
            }
        }
        block.sync();
        
        __shared__ cuda::barrier<cuda::thread_scope_block> barrier;
        cuda::memcpy_async(block, &shared_data[threadIdx.x], &input[tid], sizeof(float), barrier);
        barrier.arrive_and_wait();
        
        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (threadIdx.x < s) {
                shared_data[threadIdx.x] += shared_data[threadIdx.x + s];
            }
            block.sync();
        }
        
        if (threadIdx.x == 0) {
            output[blockIdx.x] = shared_data[0];
        }
    }
}

// Kernel 6: Convolution with shared memory - CORRECT implementation
__global__ void convolution1D(const float* __restrict__ input, const float* __restrict__ kernel, 
                               float* __restrict__ output, int n, int kernelSize) {
    __shared__ float shared_input[256 + 8];
    __shared__ cuda::barrier<cuda::thread_scope_block> barrier;
    
    auto block = cg::this_thread_block();
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int sharedIdx = threadIdx.x + kernelSize / 2;
    
    if (threadIdx.x == 0) {
        init(&barrier, blockDim.x);
    }
    block.sync();
    
    if (tid < n) {
        cuda::memcpy_async(block, &shared_input[sharedIdx], &input[tid], sizeof(float), barrier);
    }
    
    barrier.arrive_and_wait();
    
    if (tid < n) {
        float sum = 0.0f;
        for (int k = 0; k < kernelSize; k++) {
            sum += shared_input[sharedIdx - kernelSize / 2 + k] * kernel[k];
        }
        output[tid] = sum;
    }
}

// Kernel 7: Double buffering pattern - Incorrect hardcoded thread count
__global__ void doubleBufferProcess(const float* __restrict__ input, float* __restrict__ output, int n) {
    __shared__ float buffer[2][256];
    __shared__ cuda::barrier<cuda::thread_scope_block> barrier;
    
    auto block = cg::this_thread_block();
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (threadIdx.x == 0) {
        // BUG: Hardcoded 128 instead of blockDim.x
        init(&barrier, 128);
    }
    block.sync();
    
    int bufIdx = 0;
    for (int i = 0; i < 2; i++) {
        if (tid + i * blockDim.x < n) {
            cuda::memcpy_async(block, &buffer[bufIdx][threadIdx.x], 
                             &input[tid + i * blockDim.x], sizeof(float), barrier);
        }
        barrier.arrive_and_wait();
        
        if (tid + i * blockDim.x < n) {
            output[tid + i * blockDim.x] = buffer[bufIdx][threadIdx.x] * 3.14f;
        }
        bufIdx = 1 - bufIdx;
    }
}
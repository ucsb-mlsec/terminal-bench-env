#include <metal_stdlib>
using namespace metal;

// Valid compute kernel with proper parameters and thread group size
kernel void process_image([[buffer(0)]] device float* input,
                          [[buffer(1)]] device float* output,
                          uint id [[thread_position_in_grid]])
{
    output[id] = input[id] * 2.0;
}

// Valid compute kernel with thread group size specified (16*16*4 = 1024, at the limit)
kernel void filter_data([[buffer(0)]] device float4* input,
                        [[buffer(1)]] device float4* output,
                        uint2 gid [[thread_position_in_grid]],
                        uint2 tid [[thread_position_in_threadgroup]])
[[threads_per_threadgroup(16, 16, 4)]]
{
    uint index = gid.y * 1024 + gid.x;
    output[index] = input[index] * 0.5;
}

// Function missing kernel qualifier
void transform_helper([[buffer(0)]] device float* data,
                      uint id [[thread_position_in_grid]])
{
    data[id] = data[id] + 1.0;
}

// Kernel with no parameters
kernel void empty_kernel()
{
    // This won't compile properly in real Metal but tests validation
}

// Kernel that exceeds thread group limit (32*32*2 = 2048 > 1024)
kernel void compute_intensive([[buffer(0)]] device float* input,
                              [[buffer(1)]] device float* output,
                              uint3 gid [[thread_position_in_grid]])
[[threads_per_threadgroup(32, 32, 2)]]
{
    uint index = gid.z * 1024 * 1024 + gid.y * 1024 + gid.x;
    output[index] = input[index] * input[index];
}

// Valid kernel with multiple buffers
kernel void blend_textures([[buffer(0)]] device const float4* texA,
                           [[buffer(1)]] device const float4* texB,
                           [[buffer(2)]] device float4* output,
                           uint id [[thread_position_in_grid]])
{
    output[id] = texA[id] * 0.5 + texB[id] * 0.5;
}

// Valid kernel with threadgroup memory
kernel void reduce_sum([[buffer(0)]] device const float* input,
                       [[buffer(1)]] device float* output,
                       threadgroup float* shared_data [[threadgroup(0)]],
                       uint tid [[thread_position_in_threadgroup]],
                       uint bid [[threadgroup_position_in_grid]])
[[threads_per_threadgroup(256, 1, 1)]]
{
    shared_data[tid] = input[bid * 256 + tid];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    if (tid == 0) {
        float sum = 0.0;
        for (uint i = 0; i < 256; i++) {
            sum += shared_data[i];
        }
        output[bid] = sum;
    }
}
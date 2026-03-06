#include <metal_stdlib>
using namespace metal;

// Hardcoded configuration constants - candidates for function constant specialization
#define KERNEL_SIZE 5
#define TILE_SIZE 16
#define MAX_ITERATIONS 12
#define ENABLE_ALPHA true
#define USE_NORMALIZATION true
#define APPLY_GAMMA_CORRECTION false
#define PROCESS_RGB_ONLY false
#define THRESHOLD_VALUE 0.5
#define BLEND_FACTOR 0.75

// Structure for passing filter weights
struct FilterWeights {
    float weights[25]; // Supports up to 5x5 kernel
};

// Main image filtering compute kernel
kernel void imageFilterKernel(
    texture2d<float, access::read> inputTexture [[texture(0)]],
    texture2d<float, access::write> outputTexture [[texture(1)]],
    constant FilterWeights& filterWeights [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]])
{
    // Check if thread is within bounds
    if (gid.x >= outputTexture.get_width() || gid.y >= outputTexture.get_height()) {
        return;
    }
    
    float4 result = float4(0.0);
    float weightSum = 0.0;
    
    // Apply convolution filter based on KERNEL_SIZE
    int halfKernel = KERNEL_SIZE / 2;
    int weightIndex = 0;
    
    for (int dy = -halfKernel; dy <= halfKernel; dy++) {
        for (int dx = -halfKernel; dx <= halfKernel; dx++) {
            int2 samplePos = int2(gid) + int2(dx, dy);
            
            // Clamp to texture boundaries
            samplePos = clamp(samplePos, int2(0), int2(inputTexture.get_width() - 1, inputTexture.get_height() - 1));
            
            float4 sample = inputTexture.read(uint2(samplePos));
            float weight = filterWeights.weights[weightIndex++];
            
            // Process based on channel flags
            if (PROCESS_RGB_ONLY) {
                result.rgb += sample.rgb * weight;
                result.a = sample.a; // Preserve original alpha
            } else {
                result += sample * weight;
            }
            
            weightSum += weight;
        }
    }
    
    // Normalize result if enabled
    if (USE_NORMALIZATION && weightSum > 0.0) {
        result /= weightSum;
    }
    
    // Apply threshold filtering
    if (length(result.rgb) < THRESHOLD_VALUE) {
        result.rgb *= BLEND_FACTOR;
    }
    
    // Apply gamma correction if enabled
    if (APPLY_GAMMA_CORRECTION) {
        result.rgb = pow(result.rgb, float3(2.2));
    }
    
    // Handle alpha channel processing
    if (!ENABLE_ALPHA) {
        result.a = 1.0;
    }
    
    // Clamp final output
    result = clamp(result, 0.0, 1.0);
    
    outputTexture.write(result, gid);
}

// Secondary kernel for iterative refinement
kernel void iterativeRefineKernel(
    texture2d<float, access::read> inputTexture [[texture(0)]],
    texture2d<float, access::write> outputTexture [[texture(1)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= outputTexture.get_width() || gid.y >= outputTexture.get_height()) {
        return;
    }
    
    float4 accumulated = float4(0.0);
    float4 center = inputTexture.read(gid);
    
    // Perform iterative sampling based on MAX_ITERATIONS
    for (int i = 0; i < MAX_ITERATIONS; i++) {
        float radius = float(i + 1) * 0.5;
        int samples = 8;
        
        for (int s = 0; s < samples; s++) {
            float angle = float(s) / float(samples) * 2.0 * M_PI_F;
            int2 offset = int2(cos(angle) * radius, sin(angle) * radius);
            int2 samplePos = int2(gid) + offset;
            
            if (samplePos.x >= 0 && samplePos.x < int(inputTexture.get_width()) &&
                samplePos.y >= 0 && samplePos.y < int(inputTexture.get_height())) {
                accumulated += inputTexture.read(uint2(samplePos));
            }
        }
    }
    
    accumulated /= float(MAX_ITERATIONS * 8);
    
    // Blend with center pixel using BLEND_FACTOR
    float4 result = mix(center, accumulated, BLEND_FACTOR);
    
    if (USE_NORMALIZATION) {
        result = normalize(result);
    }
    
    outputTexture.write(result, gid);
}

// Tiled processing kernel for efficient memory access
kernel void tiledProcessKernel(
    texture2d<float, access::read> inputTexture [[texture(0)]],
    texture2d<float, access::write> outputTexture [[texture(1)]],
    uint2 gid [[thread_position_in_grid]],
    uint2 tid [[thread_position_in_threadgroup]])
{
    // Use TILE_SIZE for shared memory optimization
    threadgroup float4 sharedMemory[TILE_SIZE][TILE_SIZE];
    
    if (gid.x >= inputTexture.get_width() || gid.y >= inputTexture.get_height()) {
        return;
    }
    
    // Load into shared memory
    sharedMemory[tid.y][tid.x] = inputTexture.read(gid);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    float4 result = sharedMemory[tid.y][tid.x];
    
    // Process within tile boundaries
    if (tid.x > 0 && tid.x < TILE_SIZE - 1 && tid.y > 0 && tid.y < TILE_SIZE - 1) {
        float4 avg = float4(0.0);
        
        for (int dy = -1; dy <= 1; dy++) {
            for (int dx = -1; dx <= 1; dx++) {
                avg += sharedMemory[tid.y + dy][tid.x + dx];
            }
        }
        
        result = avg / 9.0;
    }
    
    outputTexture.write(result, gid);
}
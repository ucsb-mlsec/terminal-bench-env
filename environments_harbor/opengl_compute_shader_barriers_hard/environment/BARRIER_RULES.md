# GLSL Compute Shader Memory Barrier Rules

## Overview

Memory barriers in GLSL compute shaders ensure proper synchronization between shader invocations. Without proper barriers, race conditions and data corruption can occur when multiple invocations access shared resources.

## 1. SHARED Memory Barriers

### Rule
Use `barrier()` or `memoryBarrierShared()` after writing to shared memory variables when subsequent reads by other invocations in the same workgroup depend on those writes.

### When Required
- After any write to `shared` variables
- Before any read from `shared` variables that another invocation may have written
- In iterative algorithms using shared memory
- In parallel reduction operations

### Correct Usage Example
```glsl
shared float temp[256];

void main() {
    uint tid = gl_LocalInvocationID.x;
    
    // Write to shared memory
    temp[tid] = data[tid];
    
    // BARRIER REQUIRED HERE
    barrier();
    
    // Read from shared memory written by other invocations
    float value = temp[255 - tid];
}
```

### Incorrect Usage (Missing Barrier)
```glsl
shared float temp[256];

void main() {
    uint tid = gl_LocalInvocationID.x;
    
    // Write to shared memory
    temp[tid] = data[tid];
    
    // MISSING BARRIER - Race condition!
    
    // Read may see uninitialized data
    float value = temp[255 - tid];
}
```

## 2. IMAGE Memory Barriers

### Rule
Use `memoryBarrierImage()` after `imageStore()` operations when subsequent `imageLoad()` operations need to read the written data, especially across different shader invocations or workgroups.

### When Required
- After `imageStore()` calls
- Before `imageLoad()` from locations that may have been written
- In multi-pass algorithms using images
- When implementing ping-pong rendering techniques

### Correct Usage Example
```glsl
layout(rgba32f, binding = 0) uniform image2D outputImage;

void main() {
    ivec2 pos = ivec2(gl_GlobalInvocationID.xy);
    
    // Write to image
    vec4 color = computeColor(pos);
    imageStore(outputImage, pos, color);
    
    // BARRIER REQUIRED HERE
    memoryBarrierImage();
    barrier();
    
    // Read from image (possibly written by another invocation)
    vec4 neighbor = imageLoad(outputImage, pos + ivec2(1, 0));
}
```

### Incorrect Usage (Missing Barrier)
```glsl
layout(rgba32f, binding = 0) uniform image2D outputImage;

void main() {
    ivec2 pos = ivec2(gl_GlobalInvocationID.xy);
    
    // Write to image
    imageStore(outputImage, pos, computeColor(pos));
    
    // MISSING BARRIER - May read stale data!
    
    // Undefined behavior
    vec4 neighbor = imageLoad(outputImage, pos + ivec2(1, 0));
}
```

## 3. BUFFER Memory Barriers

### Rule
Use `memoryBarrierBuffer()` after writing to SSBOs (Shader Storage Buffer Objects) when subsequent reads depend on those writes, particularly when accessing via `buffer[]` or other buffer variables.

### When Required
- After writes to SSBO variables
- Before reads from SSBO locations that may have been written
- In algorithms with inter-invocation communication via buffers
- When implementing atomic operations followed by regular reads

### Correct Usage Example
```glsl
layout(std430, binding = 0) buffer OutputBuffer {
    float results[];
};

void main() {
    uint gid = gl_GlobalInvocationID.x;
    
    // Write to buffer
    results[gid] = compute(gid);
    
    // BARRIER REQUIRED HERE
    memoryBarrierBuffer();
    barrier();
    
    // Read buffer data written by other invocations
    float sum = results[gid] + results[gid + 1];
}
```

### Incorrect Usage (Missing Barrier)
```glsl
layout(std430, binding = 0) buffer Data {
    int values[];
};

void main() {
    uint gid = gl_GlobalInvocationID.x;
    
    // Write to buffer
    values[gid] = int(gid * 2);
    
    // MISSING BARRIER - Race condition!
    
    // May read incorrect data
    int neighbor = values[gid + 1];
}
```

## 4. GROUP Memory Barriers

### Rule
Use `groupMemoryBarrier()` to provide both execution and memory barriers for the current workgroup, combining effects of `barrier()` with memory synchronization across all memory types (shared, image, buffer).

### When Required
- When you need synchronization across multiple memory types
- As a comprehensive barrier in complex algorithms
- When both execution and memory ordering are critical

### Correct Usage Example
```glsl
shared float sharedData[64];
layout(std430, binding = 0) buffer Output { float out[]; };

void main() {
    uint tid = gl_LocalInvocationID.x;
    
    // Write to both shared and buffer memory
    sharedData[tid] = input[tid];
    out[tid] = input[tid] * 2.0;
    
    // BARRIER REQUIRED HERE - synchronizes all memory types
    groupMemoryBarrier();
    
    // Safe to read
    float shared_val = sharedData[(tid + 1) % 64];
    float buffer_val = out[(tid + 1) % 64];
}
```

## 5. Critical Synchronization Patterns

### Pattern 1: Parallel Reduction
```glsl
shared float reductionData[256];

// REQUIRES barrier after each reduction step
for (uint stride = 128; stride > 0; stride >>= 1) {
    if (tid < stride) {
        reductionData[tid] += reductionData[tid + stride];
    }
    barrier(); // REQUIRED after each write before next read
}
```

### Pattern 2: Iterative Algorithms
```glsl
for (int iter = 0; iter < MAX_ITER; iter++) {
    // Write phase
    imageStore(image, pos, newValue);
    
    memoryBarrierImage(); // REQUIRED
    barrier();            // REQUIRED
    
    // Read phase
    vec4 data = imageLoad(image, neighborPos);
}
```

### Pattern 3: Data Exchange Between Invocations
```glsl
// Invocation A writes
if (gl_LocalInvocationID.x == 0) {
    sharedData[0] = computeValue();
}

barrier(); // REQUIRED - ensures write is visible

// Invocation B reads
if (gl_LocalInvocationID.x == 1) {
    float val = sharedData[0];
}
```

## Summary Checklist

A barrier is REQUIRED when:
- ✓ Writing to shared memory before other invocations read it
- ✓ Using imageStore() before imageLoad() on potentially same locations
- ✓ Writing to SSBOs before reading back in same shader
- ✓ Implementing iterative or multi-pass algorithms
- ✓ Performing parallel reductions or prefix sums
- ✓ Exchanging data between invocations in a workgroup

A barrier is NOT required when:
- ✗ Each invocation only accesses its own private data
- ✗ Only reading from input buffers/images (no writes)
- ✗ Writing to completely independent memory locations with no subsequent reads
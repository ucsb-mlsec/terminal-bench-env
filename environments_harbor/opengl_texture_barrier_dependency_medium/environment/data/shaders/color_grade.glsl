#version 450

layout(local_size_x = 16, local_size_y = 16) in;

layout(binding = 4, rgba32f) uniform image2D colorBuffer;

void main() {
    ivec2 coords = ivec2(gl_GlobalInvocationID.xy);
    
    // First write
    vec4 color = vec4(0.8, 0.2, 0.4, 1.0);
    imageStore(colorBuffer, coords, color);
    
    // Proper barrier to ensure write completes
    memoryBarrierImage();
    barrier();
    
    // Now safe to read from same texture
    vec4 storedColor = imageLoad(colorBuffer, coords);
    vec4 graded = pow(storedColor, vec4(2.2));
    
    imageStore(colorBuffer, coords, graded);
}
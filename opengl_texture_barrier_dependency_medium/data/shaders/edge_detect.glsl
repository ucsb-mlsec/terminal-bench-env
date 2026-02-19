#version 450

layout(local_size_x = 8, local_size_y = 8) in;

layout(binding = 2, rgba32f) uniform image2D edgeTexture;

void main() {
    ivec2 coords = ivec2(gl_GlobalInvocationID.xy);
    
    // Multiple writes
    vec4 initial = vec4(0.0);
    imageStore(edgeTexture, coords, initial);
    imageStore(edgeTexture, coords + ivec2(1, 0), initial);
    
    // Multiple reads from same texture (VIOLATION!)
    vec4 center = imageLoad(edgeTexture, coords);
    vec4 right = imageLoad(edgeTexture, coords + ivec2(1, 0));
    vec4 bottom = imageLoad(edgeTexture, coords + ivec2(0, 1));
    vec4 left = imageLoad(edgeTexture, coords - ivec2(1, 0));
    
    // Edge detection calculation
    vec4 edge = abs(center - right) + abs(center - bottom) + abs(center - left);
    
    // Final write
    imageStore(edgeTexture, coords, edge);
}
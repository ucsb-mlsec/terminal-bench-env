#version 450

layout(local_size_x = 16, local_size_y = 16) in;

layout(binding = 0, rgba32f) uniform image2D workTexture;

void main() {
    ivec2 coords = ivec2(gl_GlobalInvocationID.xy);
    
    // First pass - write processed data
    vec4 color = vec4(1.0, 0.5, 0.3, 1.0);
    imageStore(workTexture, coords, color);
    
    // Second pass - read from same texture (VIOLATION!)
    vec4 neighbor = imageLoad(workTexture, coords + ivec2(1, 0));
    vec4 blurred = (color + neighbor) * 0.5;
    
    imageStore(workTexture, coords, blurred);
}
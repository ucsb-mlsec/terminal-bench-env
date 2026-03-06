#version 450

layout(local_size_x = 32, local_size_y = 32) in;

layout(binding = 3, rgba32f) uniform readonly image2D sourceImage;

shared vec4 sharedData[32][32];

void main() {
    ivec2 coords = ivec2(gl_GlobalInvocationID.xy);
    ivec2 localCoords = ivec2(gl_LocalInvocationID.xy);
    
    // Only reading, no writes to textures
    vec4 color = imageLoad(sourceImage, coords);
    
    // Store in shared memory (not texture memory)
    sharedData[localCoords.x][localCoords.y] = color;
}
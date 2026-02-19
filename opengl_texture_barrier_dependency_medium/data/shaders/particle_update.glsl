#version 450

layout(local_size_x = 64) in;

layout(binding = 5, rgba32f) uniform image2D particleData;

void main() {
    uint id = gl_GlobalInvocationID.x;
    ivec2 coords = ivec2(id, 0);
    
    // Read particle position
    vec4 position = imageLoad(particleData, coords);
    
    // Update position
    position.xyz += vec3(0.1, 0.0, 0.0);
    
    // Write back to same texture (VIOLATION!)
    imageStore(particleData, coords, position);
    
    // Read again from same texture (continuing VIOLATION!)
    vec4 updated = imageLoad(particleData, coords);
}
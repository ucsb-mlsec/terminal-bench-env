#version 450

layout(local_size_x = 16, local_size_y = 16) in;

layout(binding = 0, rgba32f) uniform readonly image2D inputImage;
layout(binding = 1, rgba32f) uniform writeonly image2D outputImage;

void main() {
    ivec2 coords = ivec2(gl_GlobalInvocationID.xy);
    
    // Read from input texture
    vec4 color = imageLoad(inputImage, coords);
    
    // Process the color
    color.rgb = color.rgb * 1.5;
    
    // Write to different output texture
    imageStore(outputImage, coords, color);
}
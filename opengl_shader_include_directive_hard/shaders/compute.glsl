#version 430

#include "effects/blur.glsl"

layout(local_size_x = 16, local_size_y = 16) in;

layout(rgba32f, binding = 0) uniform image2D inputImage;
layout(rgba32f, binding = 1) uniform image2D outputImage;

uniform vec2 imageSize;
uniform float blurRadius;

void main() {
    ivec2 pixelCoords = ivec2(gl_GlobalInvocationID.xy);
    
    if (pixelCoords.x >= imageSize.x || pixelCoords.y >= imageSize.y) {
        return;
    }
    
    vec4 blurredColor = applyBlur(pixelCoords, blurRadius);
    imageStore(outputImage, pixelCoords, blurredColor);
}
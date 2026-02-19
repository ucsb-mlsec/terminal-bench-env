#ifndef BLOOM_GLSL
#define BLOOM_GLSL

#include "compute.glsl"

// Bloom effect parameters
uniform float bloomThreshold;
uniform float bloomIntensity;

// Extract bright pixels for bloom
vec3 extractBrightness(vec3 color) {
    float brightness = dot(color, vec3(0.2126, 0.7152, 0.0722));
    if (brightness > bloomThreshold) {
        return color * bloomIntensity;
    }
    return vec3(0.0);
}

// Combine bloom with original scene
vec3 applyBloom(vec3 sceneColor, vec3 bloomColor) {
    return sceneColor + bloomColor * bloomIntensity;
}

#endif
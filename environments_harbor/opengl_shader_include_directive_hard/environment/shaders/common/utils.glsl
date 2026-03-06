// Common utility functions for GLSL shaders
// This file provides frequently used helper functions across the shader pipeline

#include "common/defines.glsl"

// Transform a position by a 4x4 matrix
vec3 transformPosition(vec3 pos, mat4 matrix) {
    vec4 transformed = matrix * vec4(pos, 1.0);
    return transformed.xyz / transformed.w;
}

// Transform a normal vector (ignores translation)
vec3 transformNormal(vec3 normal, mat4 matrix) {
    return normalize(mat3(matrix) * normal);
}

// Safe clamp function with epsilon handling
float safeClamp(float value, float minVal, float maxVal) {
    return clamp(value, minVal + EPSILON, maxVal - EPSILON);
}

// Remap a value from one range to another
float remap(float value, float inMin, float inMax, float outMin, float outMax) {
    float t = (value - inMin) / (inMax - inMin);
    return mix(outMin, outMax, t);
}

// Calculate luminance from RGB color
float luminance(vec3 color) {
    return dot(color, vec3(0.2126, 0.7152, 0.0722));
}

// Smooth step with configurable edge softness
float smootherstep(float edge0, float edge1, float x) {
    float t = safeClamp((x - edge0) / (edge1 - edge0), 0.0, 1.0);
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0);
}
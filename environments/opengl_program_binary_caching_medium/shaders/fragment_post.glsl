#version 330 core

// Input from vertex shader
in vec2 TexCoords;

// Output color
out vec4 FragColor;

// Texture samplers
uniform sampler2D sceneTexture;

// Post-processing parameters
uniform float brightness;
uniform float contrast;
uniform float saturation;
uniform float vignetteStrength;
uniform float blurRadius;

// Vignette effect calculation
float vignette(vec2 uv, float strength) {
    vec2 center = uv - 0.5;
    float dist = length(center);
    return 1.0 - smoothstep(0.3, 0.8, dist) * strength;
}

// Simple box blur
vec3 boxBlur(sampler2D tex, vec2 uv, float radius) {
    vec3 result = vec3(0.0);
    float count = 0.0;
    for (float x = -radius; x <= radius; x += 1.0) {
        for (float y = -radius; y <= radius; y += 1.0) {
            vec2 offset = vec2(x, y) * 0.002;
            result += texture(tex, uv + offset).rgb;
            count += 1.0;
        }
    }
    return result / count;
}

void main() {
    // Sample base color with optional blur
    vec3 color = (blurRadius > 0.0) ? boxBlur(sceneTexture, TexCoords, blurRadius) : texture(sceneTexture, TexCoords).rgb;
    
    // Apply brightness
    color *= brightness;
    
    // Apply contrast
    color = (color - 0.5) * contrast + 0.5;
    
    // Apply saturation
    float luminance = dot(color, vec3(0.299, 0.587, 0.114));
    color = mix(vec3(luminance), color, saturation);
    
    // Apply vignette effect
    float vig = vignette(TexCoords, vignetteStrength);
    color *= vig;
    
    // Clamp to valid range
    color = clamp(color, 0.0, 1.0);
    
    FragColor = vec4(color, 1.0);
}
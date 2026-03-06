#version 450 core

// Enable bindless texture extension
#extension GL_ARB_bindless_texture : require

// Bindless texture handles - must be uvec2 for ARB_bindless_texture
uniform uvec2 sceneTexture;
uniform uvec2 bloomTexture;

// Post-processing parameters
uniform float bloomStrength;
uniform float exposure;

// Input from vertex shader
in vec2 texCoord;

// Final output color
out vec4 fragColor;

// Tone mapping function
vec3 ACESFilm(vec3 x) {
    float a = 2.51;
    float b = 0.03;
    float c = 2.43;
    float d = 0.59;
    float e = 0.14;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0);
}

void main() {
    // Construct sampler2D from bindless handles
    // This is the correct way to use bindless textures in GLSL
    sampler2D sceneSampler = sampler2D(sceneTexture);
    sampler2D bloomSampler = sampler2D(bloomTexture);
    
    // Sample from both bindless textures
    vec3 sceneColor = texture(sceneSampler, texCoord).rgb;
    vec3 bloomColor = texture(bloomSampler, texCoord).rgb;
    
    // Combine scene with bloom effect
    vec3 combined = sceneColor + bloomColor * bloomStrength;
    
    // Apply exposure adjustment
    combined *= exposure;
    
    // Apply tone mapping for HDR content
    vec3 toneMapped = ACESFilm(combined);
    
    // Gamma correction
    toneMapped = pow(toneMapped, vec3(1.0 / 2.2));
    
    // Output final color
    fragColor = vec4(toneMapped, 1.0);
}
#version 450 core
#extension GL_ARB_bindless_texture : require

// Terrain Fragment Shader - Hybrid Texture Approach
// This shader mixes traditional and bindless texture sampling
// WARNING: This is a transitional implementation during migration

// Traditional sampler uniforms (old approach)
uniform sampler2D heightMap;
uniform sampler2D splatMap;

// Bindless texture handle (new approach)
uniform uvec2 terrainAlbedo;

// Material properties
uniform vec3 sunDirection;
uniform vec3 sunColor;
uniform float ambientStrength;

// Input from vertex shader
in vec2 texCoord;
in vec3 worldNormal;
in vec3 worldPosition;

// Output color
out vec4 fragColor;

void main()
{
    // Sample traditional textures using old method
    float heightValue = texture(heightMap, texCoord).r;
    vec4 splatWeights = texture(splatMap, texCoord);
    
    // Normalize splat weights
    splatWeights /= (splatWeights.r + splatWeights.g + splatWeights.b + splatWeights.a);
    
    // Convert bindless handle to sampler (new approach)
    sampler2D albedoSampler = sampler2D(terrainAlbedo);
    vec4 albedoColor = texture(albedoSampler, texCoord * 4.0);
    
    // Calculate lighting
    vec3 normal = normalize(worldNormal);
    float diffuse = max(dot(normal, sunDirection), 0.0);
    
    // Ambient component
    vec3 ambient = ambientStrength * sunColor;
    
    // Diffuse component
    vec3 diffuseLight = diffuse * sunColor;
    
    // Combine lighting with albedo
    vec3 litColor = albedoColor.rgb * (ambient + diffuseLight);
    
    // Apply height-based darkening
    litColor *= (0.7 + 0.3 * heightValue);
    
    // Apply splat map blending
    // Red channel: grass, Green: rock, Blue: sand, Alpha: snow
    vec3 finalColor = litColor;
    
    // Blend based on splat weights
    if (splatWeights.r > 0.5) {
        finalColor *= vec3(0.4, 0.6, 0.3); // Grass tint
    }
    if (splatWeights.g > 0.5) {
        finalColor *= vec3(0.5, 0.5, 0.5); // Rock tint
    }
    
    // Output final color
    fragColor = vec4(finalColor, 1.0);
}
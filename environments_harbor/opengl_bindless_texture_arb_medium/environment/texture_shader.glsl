#version 450 core

// Fragment shader for PBR material rendering
// Implements basic physically-based rendering with multiple texture maps
// Supports diffuse, normal, metallic-roughness, and ambient occlusion textures

// Extension declarations
#extension GL_ARB_separate_shader_objects : enable

// Input from vertex shader
layout(location = 0) in vec3 fragPosition;
layout(location = 1) in vec3 fragNormal;
layout(location = 2) in vec2 fragTexCoord;
layout(location = 3) in vec3 fragTangent;

// Output color
layout(location = 0) out vec4 outColor;

// Uniform buffer for material properties
layout(binding = 0) uniform MaterialUBO {
    vec4 baseColorFactor;
    float metallicFactor;
    float roughnessFactor;
    float normalScale;
    float occlusionStrength;
} material;

// Texture samplers
layout(binding = 1) uniform sampler2D albedoMap;
layout(binding = 2) uniform sampler2D normalMap;
layout(binding = 3) uniform sampler2D metallicRoughnessMap;
layout(binding = 4) uniform sampler2D aoMap;

// Light properties
layout(binding = 5) uniform LightUBO {
    vec3 lightPosition;
    vec3 lightColor;
    float lightIntensity;
} light;

void main() {
    // Sample textures
    vec4 albedo = texture(albedoMap, fragTexCoord) * material.baseColorFactor;
    vec3 normal = texture(normalMap, fragTexCoord).rgb;
    vec2 metallicRoughness = texture(metallicRoughnessMap, fragTexCoord).bg;
    float ao = texture(aoMap, fragTexCoord).r;
    
    // Convert normal from tangent space to world space
    normal = normalize(normal * 2.0 - 1.0);
    vec3 N = normalize(fragNormal);
    vec3 T = normalize(fragTangent);
    vec3 B = cross(N, T);
    mat3 TBN = mat3(T, B, N);
    N = normalize(TBN * normal);
    
    // Calculate lighting
    vec3 L = normalize(light.lightPosition - fragPosition);
    float NdotL = max(dot(N, L), 0.0);
    
    // Apply PBR factors
    float metallic = metallicRoughness.r * material.metallicFactor;
    float roughness = metallicRoughness.g * material.roughnessFactor;
    
    // Simple lighting calculation
    vec3 diffuse = albedo.rgb * light.lightColor * light.lightIntensity * NdotL;
    vec3 ambient = albedo.rgb * 0.03 * ao * material.occlusionStrength;
    
    // Final color output
    outColor = vec4(diffuse + ambient, albedo.a);
}
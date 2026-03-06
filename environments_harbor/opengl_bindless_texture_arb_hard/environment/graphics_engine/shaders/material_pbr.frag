#version 450 core

// Legacy PBR shader using traditional sampler2D uniforms
// TODO: Migrate to bindless texture system

// Traditional texture uniforms - incompatible with bindless textures
uniform sampler2D albedoMap;
uniform sampler2D normalMap;
uniform sampler2D roughnessMap;
uniform sampler2D metallicMap;
uniform sampler2D aoMap;

// Input from vertex shader
in vec2 texCoord;
in vec3 worldPos;
in vec3 worldNormal;
in mat3 TBN;

// Output
out vec4 fragColor;

// Lighting uniforms
uniform vec3 lightPos;
uniform vec3 viewPos;
uniform vec3 lightColor;

const float PI = 3.14159265359;

// Simple Fresnel-Schlick approximation
vec3 fresnelSchlick(float cosTheta, vec3 F0)
{
    return F0 + (1.0 - F0) * pow(1.0 - cosTheta, 5.0);
}

void main()
{
    // Sample textures using legacy texture sampling
    vec3 albedo = texture(albedoMap, texCoord).rgb;
    vec3 normalTex = texture(normalMap, texCoord).rgb;
    float roughness = texture(roughnessMap, texCoord).r;
    float metallic = texture(metallicMap, texCoord).r;
    float ao = texture(aoMap, texCoord).r;
    
    // Transform normal from tangent space to world space
    vec3 N = normalize(TBN * (normalTex * 2.0 - 1.0));
    vec3 V = normalize(viewPos - worldPos);
    vec3 L = normalize(lightPos - worldPos);
    vec3 H = normalize(V + L);
    
    // Basic PBR calculations
    float NdotL = max(dot(N, L), 0.0);
    float NdotH = max(dot(N, H), 0.0);
    float VdotH = max(dot(V, H), 0.0);
    
    vec3 F0 = vec3(0.04);
    F0 = mix(F0, albedo, metallic);
    vec3 F = fresnelSchlick(VdotH, F0);
    
    // Simple lighting model
    vec3 kD = (vec3(1.0) - F) * (1.0 - metallic);
    vec3 diffuse = kD * albedo / PI;
    
    vec3 specular = F * pow(NdotH, (2.0 / (roughness * roughness)) - 2.0);
    
    vec3 radiance = lightColor * NdotL;
    vec3 Lo = (diffuse + specular) * radiance * ao;
    
    // Ambient term
    vec3 ambient = vec3(0.03) * albedo * ao;
    vec3 color = ambient + Lo;
    
    fragColor = vec4(color, 1.0);
}
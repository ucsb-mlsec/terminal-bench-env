#version 400
#extension GL_ARB_gpu_shader5 : enable

// Multi-pass deferred rendering shader
// This shader outputs to multiple g-buffers for deferred shading

in vec2 vTexCoord;
in vec3 vNormal;
in vec3 vWorldPos;

uniform sampler2D albedoMap;
uniform sampler2D normalMap;
uniform sampler2D roughnessMap;

out vec4 gBuffer0;
invariant out vec4 gBuffer1;
out vec4 gBuffer2;

void main()
{
    // Sample textures
    vec4 albedo = texture(albedoMap, vTexCoord);
    vec3 normal = texture(normalMap, vTexCoord).xyz * 2.0 - 1.0;
    float roughness = texture(roughnessMap, vTexCoord).r;
    
    // Normalize the normal vector
    normal = normalize(normal);
    vec3 worldNormal = normalize(vNormal);
    
    // Output to g-buffers
    // gBuffer0: Albedo + metallic
    gBuffer0 = vec4(albedo.rgb, 0.0);
    
    // gBuffer1: Normal data
    gBuffer1 = vec4(worldNormal * 0.5 + 0.5, 1.0);
    
    // gBuffer2: Roughness + AO
    gBuffer2 = vec4(roughness, 1.0, 0.0, 1.0);
}
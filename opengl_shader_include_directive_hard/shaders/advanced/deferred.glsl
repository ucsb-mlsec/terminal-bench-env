#version 450 core

// Deferred rendering shader - G-buffer pass
#include "gbuffer/layout.glsl"
#include "lighting/pbr.glsl"
#include "common/defines.glsl"

layout(location = 0) in vec3 fragPosition;
layout(location = 1) in vec3 fragNormal;
layout(location = 2) in vec2 fragTexCoord;
layout(location = 3) in vec3 fragTangent;

layout(location = 0) out vec4 gPosition;
layout(location = 1) out vec4 gNormal;
layout(location = 2) out vec4 gAlbedoSpec;
layout(location = 3) out vec4 gMaterial;

uniform sampler2D albedoMap;
uniform sampler2D normalMap;
uniform sampler2D metallicMap;
uniform sampler2D roughnessMap;

void main() {
    // Store fragment position in world space
    gPosition = vec4(fragPosition, 1.0);
    
    // Store normal information
    vec3 normal = normalize(fragNormal);
    gNormal = vec4(normal, 1.0);
    
    // Store albedo and specular
    vec3 albedo = texture(albedoMap, fragTexCoord).rgb;
    float specular = texture(metallicMap, fragTexCoord).r;
    gAlbedoSpec = vec4(albedo, specular);
    
    // Store material properties
    float metallic = texture(metallicMap, fragTexCoord).r;
    float roughness = texture(roughnessMap, fragTexCoord).r;
    gMaterial = vec4(metallic, roughness, 0.0, 1.0);
}
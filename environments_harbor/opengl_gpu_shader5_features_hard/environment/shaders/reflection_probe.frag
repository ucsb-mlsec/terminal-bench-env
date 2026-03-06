#version 150
#extension GL_ARB_gpu_shader5 : enable

in vec3 worldPos;
in vec3 normal;
in vec3 viewDir;

out vec4 fragColor;

uniform samplerCube envMap;
uniform samplerCube irradianceMap;
uniform float roughness;
uniform float metallic;

void main()
{
    vec3 N = normalize(normal);
    vec3 V = normalize(viewDir);
    vec3 R = reflect(-V, N);
    
    // Sample environment maps
    vec3 envColor = textureLod(envMap, R, roughness * 8.0).rgb;
    vec3 irradiance = texture(irradianceMap, N).rgb;
    
    // Calculate fresnel term
    float fresnel = pow(1.0 - max(dot(N, V), 0.0), 5.0);
    fresnel = 0.04 + (1.0 - 0.04) * fresnel;
    
    // Calculate diffuse component
    vec3 albedo = vec3(0.8, 0.8, 0.8);
    vec3 diffuse = albedo * irradiance * (1.0 - metallic);
    
    // Blend reflection using FMA
    vec3 reflection = fma(envColor, vec3(fresnel), vec3(0.0));
    
    // Combine reflection and diffuse using FMA
    vec3 combined = fma(reflection, vec3(metallic), diffuse);
    
    fragColor = vec4(combined, 1.0);
}
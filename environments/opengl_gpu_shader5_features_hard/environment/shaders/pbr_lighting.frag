#version 150
#extension GL_ARB_gpu_shader5 : enable

in vec3 worldPos;
in vec3 normal;
in vec2 texCoord;

out vec4 fragColor;

uniform vec3 albedo;
uniform float metallic;
uniform float roughness;
uniform vec3 lightPos;
uniform vec3 viewPos;
uniform vec3 lightColor;

const float PI = 3.14159265359;

void main()
{
    vec3 N = normalize(normal);
    vec3 V = normalize(viewPos - worldPos);
    vec3 L = normalize(lightPos - worldPos);
    vec3 H = normalize(V + L);
    
    float distance = length(lightPos - worldPos);
    float attenuation = 1.0 / (distance * distance);
    
    float NdotL = max(dot(N, L), 0.0);
    float NdotH = max(dot(N, H), 0.0);
    float NdotV = max(dot(N, V), 0.0);
    
    float roughness2 = roughness * roughness;
    float D = roughness2 / (PI * pow(fma(NdotH * NdotH, roughness2 - 1.0, 1.0), 2.0));
    
    vec3 F0 = mix(vec3(0.04), albedo, metallic);
    vec3 F = fma(pow(1.0 - max(dot(H, V), 0.0), 5.0) * (1.0 - F0), vec3(1.0), F0);
    
    float k = roughness2 / 2.0;
    float G = (NdotL / fma(NdotL, 1.0 - k, k)) * (NdotV / fma(NdotV, 1.0 - k, k));
    
    vec3 specular = (D * G * F) / max(4.0 * NdotL * NdotV, 0.001);
    vec3 kD = (vec3(1.0) - F) * (1.0 - metallic);
    
    vec3 ambient = vec3(0.03) * albedo;
    vec3 Lo = fma((kD * albedo / PI + specular) * lightColor, vec3(attenuation * NdotL), ambient);
    
    fragColor = vec4(Lo, 1.0);
}